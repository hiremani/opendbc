import math
import unittest

import numpy as np

from opendbc.can import CANPacker
from opendbc.car import Bus, structs
from opendbc.car.fisker.carcontroller import (CarController, LAT_CANDIDATE_ADDR, LAT_CANDIDATE_BUS,
                                              apply_fisker_steer_angle_limits)
from opendbc.car.fisker.carstate import CarState
from opendbc.car.fisker.interface import CarInterface
from opendbc.car.fisker.values import (ANGLE_OVERRIDE_DEG, ANGLE_OVERRIDE_RELEASE_DEG, CANBUS, CAR,
                                        DBC, OVERRIDE_RELEASE_FRAMES, OVERRIDE_RELEASE_TORQUE,
                                        OVERRIDE_STEER_TORQUE, OVERRIDE_TRIP_FRAMES)
from opendbc.car.lateral import apply_std_steer_angle_limits
from opendbc.car.fisker.values import CarControllerParams, LAT_ANGLE_CMD_MAX
from opendbc.testing import parameterized


def candidate_angle(frames):
  """The 0x1D0 angle carried by this tick's candidate window (every candidate carries the same one)."""
  angles = {int.from_bytes(data[2:4], 'big') * .0625 - 780
            for addr, data, bus in frames if addr == LAT_CANDIDATE_ADDR and bus == LAT_CANDIDATE_BUS}
  assert len(angles) == 1
  return angles.pop()


def _make_car():
  cp = CarInterface.get_non_essential_params(CAR.FISKER_OCEAN)
  cp.secOcKeyAvailable = True
  sp = CarInterface.get_non_essential_params_sp(cp, CAR.FISKER_OCEAN)
  cs = CarState(cp, sp)
  return cp, sp, cs


def candidate_wires(frames):
  """SecOC wire counters ((byte4 >> 2) & 0x3F) of this tick's 0x1D0 candidates."""
  return sorted({(data[4] >> 2) & 0x3F for addr, data, bus in frames
                 if addr == LAT_CANDIDATE_ADDR and bus == LAT_CANDIDATE_BUS})


def _cartesian(*sets):
  """Every combination of the parameter sets, flattened into one tuple each."""
  combos = [()]
  for values in sets:
    combos = [c + (v if isinstance(v, tuple) else (v,)) for c in combos for v in values]
  return combos


class TestSteeringHandover(unittest.TestCase):
  @parameterized('cruise, last, measured, speed', _cartesian([False, True], [(-1.8125, 3.3125, 18.52), (18.875, 11.6875, 18.78), (46.875, 41.25, 10.4)]))
  def test_override_releases_immediately_without_angle_jump(self, last, measured, speed, cruise):
    car = _make_car()
    cp, sp, cs = car
    controller = CarController(DBC[CAR.FISKER_OCEAN], cp, sp)
    controller.secoc_key_verified = True
    controller.secoc_key = bytes(16)  # synthetic key, no vehicle credentials
    controller.apply_angle_last = last
    controller.was_lat_engaged_prev = True
    controller.was_tx_lateral_prev = True  # mid-session: we were already transmitting
    cs.out = structs.CarState(steeringAngleDeg=measured, vEgoRaw=speed)
    cs.out.cruiseState.enabled = cruise
    cs.driver_intervening = True
    cc = structs.CarControl(latActive=True)
    cc.actuators.steeringAngleDeg = last
    cc_sp = structs.CarControlSP()
    cc_sp.mads.active = True
    _, frames = controller.update(cc.as_reader(), cc_sp, cs, 0)
    messages = {addr: data for addr, data, bus in frames if bus == CANBUS.pt}
    assert (messages[0x1C0][3] >> 6) & 3 == 0
    assert messages[0x1C0][2] & 32  # override remains signaled
    assert 0x1D0 not in messages  # the angle reaches the EPS only through panda's candidate substitution
    sent_angle = candidate_angle(frames)
    assert abs(sent_angle - last) < 2.0
    assert abs(sent_angle - measured) > 3

  @parameterized('last, target, speed', _cartesian([(-20, 20), (20, -20), (0, 900), (0, -900)], [0, 5, 10, 25, 40]))
  def test_active_limits_unchanged(self, speed, last, target):
    assert apply_fisker_steer_angle_limits(target, last, speed, 0, True, True) == apply_std_steer_angle_limits(
      target, last, speed, 0, True, CarControllerParams.ANGLE_LIMITS)

  @parameterized('sign', [1, -1])
  def test_command_never_near_full_lock(self, sign):
    # route 00000103 (2026-11-16): the model asked for 460 deg at 5 mph and the EPS aborted (AbortFb=9)
    last = sign * 380.0
    for _ in range(200):
      last = apply_fisker_steer_angle_limits(sign * 460.0, last, 2.3, sign * 445.0, True, True)
      assert abs(last) <= LAT_ANGLE_CMD_MAX
    assert last == sign * LAT_ANGLE_CMD_MAX

  def test_not_steering_follows_the_wheel_past_the_cap(self):
    # the driver's own full-lock turn: the heartbeat keeps following the measured wheel (panda checks it)
    last = 440.0
    last = apply_fisker_steer_angle_limits(0.0, last, 1.0, 466.0, False, True)
    assert last > LAT_ANGLE_CMD_MAX
    assert apply_fisker_steer_angle_limits(0.0, 0.0, 1.0, 466.0, False, False) == 466.0

  def test_disabled_heartbeat_matches_measured_angle(self):
    assert apply_fisker_steer_angle_limits(50, 20, 15, -10, False, False) == -10

  @parameterized('cruise', [False, True])
  def test_disengagement_fade_respects_remaining_permission(self, cruise):
    car = _make_car()
    cp, sp, cs = car
    controller = CarController(DBC[CAR.FISKER_OCEAN], cp, sp)
    controller.secoc_key_verified = True
    controller.secoc_key = bytes(16)
    controller.apply_angle_last = 20
    controller.was_lat_engaged_prev = True
    controller.was_tx_lateral_prev = True  # the fade continues a transmitting session
    controller.lat_fade_remaining = controller.FADE_TICKS
    cs.out = structs.CarState(steeringAngleDeg=-10, vEgoRaw=15)
    cs.out.cruiseState.enabled = cruise
    cc = structs.CarControl(latActive=False)
    _, frames = controller.update(cc.as_reader(), structs.CarControlSP(), cs, 0)
    messages = {addr: data for addr, data, bus in frames if bus == CANBUS.pt}
    assert (messages[0x1C0][3] >> 6) & 3 == 0
    angle = candidate_angle(frames)
    if cruise:
      assert 18 < angle < 20  # ACC still leaves Panda's active rate check enabled
    else:
      assert angle == -10  # no permission: inactive check requires measured angle

  def test_fault_requires_valid_eps_recovery(self):
    car = _make_car()
    cp, sp, cs = car
    parsers = cs.get_can_parsers(cp, sp)
    packer = CANPacker(DBC[CAR.FISKER_OCEAN][Bus.pt])

    def update(state, validity, abort, field='steerFaultTemporary'):
      frame = packer.make_can_msg('EPS_0x1C2', CANBUS.pt, {
        'EPS_AdasLatCtrlSts': state, 'EPS_AdasLatCtrlStsVld': validity, 'EPS_AbortFb': abort})
      for parser in parsers.values():
        parser.update([(0, [frame])])
      return getattr(cs.update(parsers)[0], field)

    assert not update(0, 0, 0)  # startup is not a historical fault
    assert not update(2, 1, 0)
    assert not update(0, 0, 1)  # expected driver intervention alone
    assert update(3, 1, 8)
    for _ in range(100):
      assert update(0, 0, 0)  # recorded false-recovery state
    assert update(1, 2, 0)  # invalid Available must not clear it
    assert update(2, 0, 0)  # initializing Active must not clear it
    assert update(1, 1, 8)  # valid Available with an abort is still faulted
    assert not update(1, 1, 0)
    assert update(3, 1, 6)
    assert not update(2, 1, 0)

    # Sustained abort escalates to a permanent fault (MADS disengage); it tracks the live
    # abort, so it clears as soon as the EPS stops reporting one, even while the soft latch holds.
    assert not update(1, 1, 8, 'steerFaultPermanent')
    assert not update(1, 1, 8, 'steerFaultPermanent')
    assert update(1, 1, 8, 'steerFaultPermanent')
    assert not update(0, 0, 0, 'steerFaultPermanent')
    assert update(0, 0, 0)  # soft latch still waiting for a valid recovery

  def test_angle_divergence_override(self):
    car = _make_car()
    # On-vehicle 2026-10-27: during a hand U-turn the EPS assisted the turn, so column torque
    # stayed under the yield threshold and the EPS intervention bit never tripped, yet the wheel
    # ran ~70 deg past our command -> openpilot kept commanding into a near-full-lock abort=9
    # (permanent LKAS fault). When the measured wheel diverges far from the commanded angle we
    # must yield. Hysteresis: latch at OVERRIDE_DEG, release within RELEASE_DEG.
    cp, sp, cs = car
    controller = CarController(DBC[CAR.FISKER_OCEAN], cp, sp)
    controller.secoc_key_verified = True
    controller.secoc_key = bytes(16)
    cc_sp = structs.CarControlSP()
    cc_sp.mads.active = True
    controller.was_lat_engaged_prev = True
    controller.was_lat_active_prev = True
    controller.engage_delay_remaining = 0
    controller.apply_angle_last = 5.0
    cs.out = structs.CarState(steeringAngleDeg=5.0, vEgoRaw=20.0)
    cs.eps_lat_sts = 2
    cc = structs.CarControl(latActive=True)
    cc.actuators.steeringAngleDeg = 5.0   # model commands 5 deg throughout

    controller.update(cc.as_reader(), cc_sp, cs, 0)
    assert not controller._angle_override_latch                      # wheel on target

    cs.out.steeringAngleDeg = 5.0 + ANGLE_OVERRIDE_DEG + 10.0         # driver turns well past
    _, frames = controller.update(cc.as_reader(), cc_sp, cs, 0)
    assert controller._angle_override_latch                          # -> yield
    # fully de-assert while latched: no candidates sent, so OEM's own 0x1D0 feeds the EPS
    assert LAT_CANDIDATE_ADDR not in {addr for addr, _, bus in frames if bus == LAT_CANDIDATE_BUS}

    cs.out.steeringAngleDeg = 5.0 + (ANGLE_OVERRIDE_DEG + ANGLE_OVERRIDE_RELEASE_DEG) / 2
    controller.update(cc.as_reader(), cc_sp, cs, 0)
    assert controller._angle_override_latch                          # held in hysteresis band

    cs.out.steeringAngleDeg = 5.0 + ANGLE_OVERRIDE_RELEASE_DEG - 2.0  # wheel back near command
    controller.update(cc.as_reader(), cc_sp, cs, 0)
    assert not controller._angle_override_latch                      # -> resume

  def test_model_target_leading_the_wheel_is_not_an_override(self):
    # On-vehicle 2026-11-14: in signaled low-speed turns the model's target ran > ANGLE_OVERRIDE_DEG ahead of
    # the wheel while the wheel was following our rate-limited command, and the latch handed every turn back to
    # the driver. Only a wheel far from what we commanded too is the driver steering.
    cp, sp, cs = _make_car()
    controller = CarController(DBC[CAR.FISKER_OCEAN], cp, sp)
    controller.secoc_key_verified = True
    controller.secoc_key = bytes(16)
    cc_sp = structs.CarControlSP()
    cc_sp.mads.active = True
    controller.was_lat_engaged_prev = True
    controller.was_lat_active_prev = True
    controller.was_tx_lateral_prev = True
    controller.engage_delay_remaining = 0
    cs.eps_lat_sts = 2
    cc = structs.CarControl(latActive=True)

    controller.apply_angle_last = -40.0                                # our command so far, walking into the turn
    cs.out = structs.CarState(steeringAngleDeg=-35.0, vEgoRaw=3.0)    # wheel following it, 5 deg behind
    cc.actuators.steeringAngleDeg = -40.0 - ANGLE_OVERRIDE_DEG - 30.0  # model's target 65 deg ahead of the wheel
    _, frames = controller.update(cc.as_reader(), cc_sp, cs, 0)
    assert not controller._angle_override_latch
    assert LAT_CANDIDATE_ADDR in {addr for addr, _, bus in frames if bus == LAT_CANDIDATE_BUS}  # still steering
    assert controller.apply_angle_last < -40.0                        # and still walking toward the target

    # the driver yanks the wheel the other way, far from both: that is an override
    cs.out.steeringAngleDeg = controller.apply_angle_last + ANGLE_OVERRIDE_DEG + 10.0
    controller.update(cc.as_reader(), cc_sp, cs, 0)
    assert controller._angle_override_latch

  def _turn_controller(self):
    cp, sp, cs = _make_car()
    controller = CarController(DBC[CAR.FISKER_OCEAN], cp, sp)
    controller.secoc_key_verified = True
    controller.secoc_key = bytes(16)
    cc_sp = structs.CarControlSP()
    cc_sp.mads.active = True
    controller.was_lat_engaged_prev = True
    controller.was_lat_active_prev = True
    controller.was_tx_lateral_prev = True
    controller.engage_delay_remaining = 0
    cs.eps_lat_sts = 2
    return controller, cc_sp, cs

  def test_driver_leading_a_signaled_turn_is_taken_over(self):
    # On-vehicle 2026-11-14: the driver starts each signaled turn before the turn planner, the latch trips, and the
    # planner's target (-263 deg) never comes within RELEASE_DEG of the held wheel (-140..-180): every turn handed
    # back. Once the command wants at least as much turn the same way, taking over just carries the turn on.
    controller, cc_sp, cs = self._turn_controller()
    cc = structs.CarControl(latActive=True)
    controller.apply_angle_last = 20.0
    controller._prev_steer_angle = -50.0
    cs.out = structs.CarState(steeringAngleDeg=-52.0, vEgoRaw=1.5, rightBlinker=True)
    cc.actuators.steeringAngleDeg = 20.0           # planner not turning yet: the command still points the other way
    controller.update(cc.as_reader(), cc_sp, cs, 0)
    assert controller._angle_override_latch        # driver ahead -> yield, as before

    for wheel in (-90.0, -138.0, -140.0):          # the driver keeps turning; the planner starts and overtakes
      cs.out.steeringAngleDeg = wheel
      cc.actuators.steeringAngleDeg = -263.0
      controller.update(cc.as_reader(), cc_sp, cs, 0)
    assert not controller._angle_override_latch    # -> taken over mid-turn
    assert controller._settle_hold == 0            # without waiting for the (turning) wheel to settle
    assert controller.engage_delay_remaining > 0   # still the normal re-grab hold from the live wheel

  def test_unwinding_or_unsignaled_or_fast_is_still_an_override(self):
    cases = [
      dict(wheel=-140.0, prev=-150.0, v=1.5, right=True, cmd=-263.0),    # driver straightening out of the turn
      dict(wheel=-140.0, prev=-139.0, v=1.5, right=False, cmd=-263.0),   # no blinker
      dict(wheel=-140.0, prev=-139.0, v=12.0, right=True, cmd=-263.0),   # above TURN_COOP_MAX_SPEED
      dict(wheel=-140.0, prev=-139.0, v=1.5, right=True, cmd=-60.0),     # command wants less turn than the wheel
    ]
    for c in cases:
      controller, cc_sp, cs = self._turn_controller()
      cc = structs.CarControl(latActive=True)
      controller.apply_angle_last = 0.0
      controller._prev_steer_angle = c["prev"]
      cs.out = structs.CarState(steeringAngleDeg=c["wheel"], vEgoRaw=c["v"], rightBlinker=c["right"])
      cc.actuators.steeringAngleDeg = c["cmd"]
      controller.update(cc.as_reader(), cc_sp, cs, 0)
      assert controller._angle_override_latch, c

  def test_torque_latch_held_by_a_turn_grip_is_set_aside(self):
    # On-vehicle 2026-11-14 (ee seg 17, t 25.7): the driver pushed 4 Nm into a right turn the car had not started,
    # the torque latch tripped, and holding the turned wheel (1.6-2.6 Nm) kept it above the 0.5 Nm release for the
    # whole turn. In a cooperative signaled turn that latch alone no longer holds the yield.
    for torque, eps_bit, expect_steer in ((-1.8, False, True),    # holding into the turn: steer
                                          (+1.0, False, False),   # pulling back toward center: yield
                                          (-3.5, False, False),   # pushing past the trip level: yield
                                          (-1.8, True, False)):   # the EPS's own overpower bit: always yield
      controller, cc_sp, cs = self._turn_controller()
      cc = structs.CarControl(latActive=True)
      controller._prev_steer_angle = -139.0
      controller.apply_angle_last = -140.0
      cs.out = structs.CarState(steeringAngleDeg=-140.0, vEgoRaw=2.0, rightBlinker=True, steeringTorque=torque)
      cs.driver_intervening = True
      cs.torque_latch_only = not eps_bit
      cc.actuators.steeringAngleDeg = -263.0
      for _ in range(controller.ENGAGE_DELAY_TICKS + 2):   # through the normal re-grab hold
        controller.update(cc.as_reader(), cc_sp, cs, 0)
      # lat_active (the Req bit); a torque pause keeps the candidate stream running with Req=0
      assert controller.was_lat_active_prev == expect_steer, (torque, eps_bit)

  def test_coop_signaled_turn_predicate(self):
    from opendbc.car.fisker.carcontroller import coop_signaled_turn
    assert coop_signaled_turn(-263.0, -140.0, -1.0, 2.0, False, True)        # right turn, turning in
    assert coop_signaled_turn(200.0, 120.0, 0.0, 2.0, True, False)           # left turn, holding
    assert not coop_signaled_turn(200.0, 120.0, 0.0, 2.0, False, True)       # blinker the other way
    assert not coop_signaled_turn(-263.0, 140.0, 0.0, 2.0, True, False)      # command the other way
    assert not coop_signaled_turn(-263.0, -140.0, 0.0, 2.0, True, True)      # hazards: no side

  def test_torque_override_yields_below_eps_threshold(self):
    car = _make_car()
    # The EPS's own intervention bit only trips at ~6.5 Nm. We yield proactively at
    # OVERRIDE_STEER_TORQUE (driver torque reads ~0 hands-off), so an override takes less
    # effort and releases before the EPS fights up to its hard threshold.
    cp, sp, cs = car
    parsers = cs.get_can_parsers(cp, sp)
    packer = CANPacker(DBC[CAR.FISKER_OCEAN][Bus.pt])

    def intervening(nm, detd=0, vld=1):
      frame = packer.make_can_msg('EPS_0x1C4', CANBUS.pt, {
        'EPS_DrvrSteerTq': abs(nm), 'EPS_DrvrSteerTqDir': 1 if nm < 0 else 0,
        'EPS_DrvrIntvSteerWhlDetd': detd, 'EPS_DrvrIntvSteerWhlVld': vld})
      for parser in parsers.values():
        parser.update([(0, [frame])])
      return cs.update(parsers)[0] is not None and cs.driver_intervening

    assert not intervening(OVERRIDE_STEER_TORQUE - 1.0)   # under threshold, EPS bit off -> hold
    # A bump: a spike just short of the trip time, then gone, never yields.
    for _ in range(OVERRIDE_TRIP_FRAMES - 1):
      assert not intervening(OVERRIDE_STEER_TORQUE + 3.0)
    assert not intervening(0.0)
    for _ in range(OVERRIDE_TRIP_FRAMES - 1):
      assert not intervening(OVERRIDE_STEER_TORQUE + 0.5)  # the count starts again after a dip
    assert intervening(OVERRIDE_STEER_TORQUE + 0.5)       # held for the trip time -> yield (latched)
    assert intervening(OVERRIDE_RELEASE_TORQUE + 0.5)     # held in the hysteresis band, no chatter
    assert intervening(-(OVERRIDE_STEER_TORQUE + 0.5))    # stays latched, either direction
    # Release only after torque stays below the release gate for the full hold time.
    released = False
    for _ in range(OVERRIDE_RELEASE_FRAMES + 2):
      released = not intervening(0.0)
    assert released
    assert intervening(0.0, detd=1)                       # EPS bit remains an independent backstop

    # The takeover torque is set live (OceanScape's slider); inf leaves only the EPS bit.
    intervening(0.0)
    cs.override_torque = 5.0
    for _ in range(OVERRIDE_TRIP_FRAMES + 2):
      assert not intervening(4.5)
    for _ in range(OVERRIDE_TRIP_FRAMES - 1):
      assert not intervening(5.5)
    assert intervening(5.5)
    for _ in range(OVERRIDE_RELEASE_FRAMES + 2):
      intervening(0.0)
    cs.override_torque = float('inf')
    for _ in range(OVERRIDE_TRIP_FRAMES + 2):
      assert not intervening(6.0)
    assert intervening(6.0, detd=1)

  def test_standstill_pauses_lateral(self):
    car = _make_car()
    # Below ~1 mph we pause actuation so the wheel doesn't creep/center at a stop: Req follows
    # the measured angle (heartbeat) instead of steering to the model target.
    cp, sp, cs = car
    controller = CarController(DBC[CAR.FISKER_OCEAN], cp, sp)
    controller.secoc_key_verified = True
    controller.secoc_key = bytes(16)
    cc_sp = structs.CarControlSP()
    cc_sp.mads.active = True
    controller.was_lat_engaged_prev = True
    controller.was_lat_active_prev = True
    controller.engage_delay_remaining = 0
    controller.apply_angle_last = 10.0
    cs.out = structs.CarState(steeringAngleDeg=10.0, vEgoRaw=0.0)   # stopped
    cs.eps_lat_sts = 2
    cc = structs.CarControl(latActive=True)
    cc.actuators.steeringAngleDeg = 30.0                           # model wants 30 deg
    _, frames = controller.update(cc.as_reader(), cc_sp, cs, 0)
    assert abs(candidate_angle(frames) - 10.0) < 2.0              # paused: holds measured, not 30

  def test_resume_from_pause_rearms_engage_delay(self):
    car = _make_car()
    # Coming out of a pause (override release or crossing the standstill speed) within an
    # engaged session re-arms the engage delay, so we ease back in instead of yanking the wheel.
    cp, sp, cs = car
    controller = CarController(DBC[CAR.FISKER_OCEAN], cp, sp)
    controller.secoc_key_verified = True
    controller.secoc_key = bytes(16)
    cc_sp = structs.CarControlSP()
    cc_sp.mads.active = True
    controller.was_lat_engaged_prev = True
    controller.was_lat_active_prev = False    # lateral was paused last tick
    controller.engage_delay_remaining = 0
    cs.out = structs.CarState(steeringAngleDeg=0.0, vEgoRaw=20.0)
    cs.eps_lat_sts = 2
    cc = structs.CarControl(latActive=True)
    controller.update(cc.as_reader(), cc_sp, cs, 0)
    assert controller.engage_delay_remaining >= controller.ENGAGE_DELAY_TICKS - 1

  def test_settle_wait_holds_until_wheel_steadies(self):
    car = _make_car()
    # On resume we must not re-grab a wheel that's still self-centering fast, or we fight it
    # (jerk + transient EPS abort + false limit alert). Keep holding while the wheel moves,
    # resume only once it steadies.
    cp, sp, cs = car
    controller = CarController(DBC[CAR.FISKER_OCEAN], cp, sp)
    controller.secoc_key_verified = True
    controller.secoc_key = bytes(16)
    cc_sp = structs.CarControlSP()
    cc_sp.mads.active = True
    controller.was_lat_engaged_prev = True
    controller.was_lat_active_prev = False    # resuming from a pause
    controller.engage_delay_remaining = 0
    cs.eps_lat_sts = 2
    cc = structs.CarControl(latActive=True)

    angle = 60.0
    for _ in range(15):                       # wheel self-centering fast (4 deg/tick)
      cs.out = structs.CarState(steeringAngleDeg=angle, vEgoRaw=20.0)
      controller.update(cc.as_reader(), cc_sp, cs, 0)
      angle -= 4.0
    assert controller._settle_hold > 0                                        # still waiting
    assert controller.engage_delay_remaining >= controller.ENGAGE_DELAY_TICKS - 1   # still held

    for _ in range(5):                        # wheel steady now
      cs.out = structs.CarState(steeringAngleDeg=8.0, vEgoRaw=20.0)
      controller.update(cc.as_reader(), cc_sp, cs, 0)
    assert controller._settle_hold == 0                                       # settled -> resume

  def test_override_eps_off_deasserts_then_rearms(self):
    car = _make_car()
    # On-vehicle 2026-10-26: a hard overpower drives the EPS to Off (sts=0) and it stays there
    # while we keep asserting Sts=1, until we stop transmitting. So on EPS-Off we must de-assert
    # (send nothing) to let it reset to Available, then re-arm a fresh engage once it's back and
    # the driver has let go — otherwise lateral pauses on the first override and never resumes.
    cp, sp, cs = car
    controller = CarController(DBC[CAR.FISKER_OCEAN], cp, sp)
    controller.secoc_key_verified = True
    controller.secoc_key = bytes(16)
    cc_sp = structs.CarControlSP()
    cc_sp.mads.active = True
    controller.was_lat_engaged_prev = True
    controller.was_lat_active_prev = True
    controller.engage_delay_remaining = 0
    cs.out = structs.CarState(steeringAngleDeg=0.0, vEgoRaw=20.0)
    cs.eps_lat_sts = 2  # Active
    cc = structs.CarControl(latActive=True)

    def candidates_present():
      _, frames = controller.update(cc.as_reader(), cc_sp, cs, 0)
      return LAT_CANDIDATE_ADDR in {addr for addr, _, bus in frames if bus == LAT_CANDIDATE_BUS}

    assert candidates_present()                      # steady state: transmitting

    # Hard overpower drops the EPS to Off -> we must stop transmitting so it can reset.
    cs.eps_lat_sts = 0
    cs.out.steeringPressed = True
    for _ in range(3):
      assert not candidates_present()
    assert controller._override_eps_recovery

    # EPS back to Available but driver still holding -> stay de-asserted (don't fight).
    cs.eps_lat_sts = 1
    assert not candidates_present()
    assert controller._override_eps_recovery

    # EPS Available AND driver released -> re-arm the engage delay for a fresh handshake.
    cs.out.steeringPressed = False
    controller.update(cc.as_reader(), cc_sp, cs, 0)
    assert not controller._override_eps_recovery
    assert controller.engage_delay_remaining == controller.ENGAGE_DELAY_TICKS

  def test_engage_handoff_seeds_from_measured_angle(self):
    car = _make_car()
    # On-vehicle 2026-10-26: at a moving engage our command was still near 0 (model target)
    # while the wheel sat at -2.4 deg, so panda dropped every candidate for ~160 ms until the
    # limiter slewed in. Through the engage-delay hold apply_angle_last must track the measured
    # wheel angle so the first transmitted frame is a legal one-step slew from the real wheel
    # position, not from a stale reference.
    cp, sp, cs = car
    controller = CarController(DBC[CAR.FISKER_OCEAN], cp, sp)
    controller.secoc_key_verified = True
    controller.secoc_key = bytes(16)
    cc_sp = structs.CarControlSP()
    cc_sp.mads.active = True
    controller.apply_angle_last = 0.0            # stale straight-ahead reference
    measured = -2.4
    cs.out = structs.CarState(steeringAngleDeg=measured, vEgoRaw=20.0)
    cc = structs.CarControl(latActive=True)
    cc.actuators.steeringAngleDeg = 0.0          # model wants straight

    # Rising edge opens the engage delay; through the hold we send nothing and
    # apply_angle_last stays glued to the measured wheel angle. The rising-edge tick seeds
    # the delay and then counts one off it, so there are ENGAGE_DELAY_TICKS - 1 pure hold
    # ticks before the first TX.
    for _ in range(controller.ENGAGE_DELAY_TICKS - 1):
      _, frames = controller.update(cc.as_reader(), cc_sp, cs, 0)
      assert LAT_CANDIDATE_ADDR not in {addr for addr, _, bus in frames if bus == LAT_CANDIDATE_BUS}
      assert math.isclose(controller.apply_angle_last, measured, abs_tol=1e-3)

    # First tick past the hold: we transmit the measured angle itself (no slew, so panda's
    # fresher wheel angle can't put it past the rate limit), never the stale 0.
    _, frames = controller.update(cc.as_reader(), cc_sp, cs, 0)
    assert math.isclose(candidate_angle(frames), measured, abs_tol=0.0625)

    # Then we slew toward the target one rate-limited step at a time from there.
    _, frames = controller.update(cc.as_reader(), cc_sp, cs, 0)
    step = candidate_angle(frames) - measured
    limit = float(np.interp(20.0, *CarControllerParams.ANGLE_LIMITS.ANGLE_RATE_LIMIT_DOWN))
    assert 0.0 < step <= limit + 0.0625

  def test_gw_reset_prestamps_next_window(self):
    car = _make_car()
    # Near the end of a GW reset window the controller must also stamp the first counters of
    # the NEXT window (reset + 1), so panda has a match when OEM restarts at 1 before carstate
    # has seen the restart — otherwise those frames drop and the EPS aborts=8.
    cp, sp, cs = car
    controller = CarController(DBC[CAR.FISKER_OCEAN], cp, sp)
    controller.secoc_key_verified = True
    controller.secoc_key = bytes(16)
    cc_sp = structs.CarControlSP()
    cc_sp.mads.active = True
    controller.was_lat_engaged_prev = True   # already engaged: past the engage-delay window
    controller.was_lat_active_prev = True
    controller.engage_delay_remaining = 0
    cs.out = structs.CarState(steeringAngleDeg=0.0, vEgoRaw=20.0)
    cs.secoc_trip, cs.secoc_reset = 1, 5
    cs.oem_1d0_reset_flag = 5 & 0x3
    cs.oem_1d0_alive, cs.oem_1c0_alive = 3, 3
    cc = structs.CarControl(latActive=True)

    # Mid-window: candidates straddle OEM's counter, none for the next window.
    cs.oem_1d0_secoc_full_ctr = 47
    _, frames = controller.update(cc.as_reader(), cc_sp, cs, 0)
    assert 47 in candidate_wires(frames)
    assert not any(w <= 11 for w in candidate_wires(frames))

    # Near the end of the window: the next window's counters 1..6, stamped for reset 6.
    cs.oem_1d0_secoc_full_ctr = 98
    _, frames = controller.update(cc.as_reader(), cc_sp, cs, 10_000_000)
    low = {(data[4] >> 2) & 0x3F: data[4] & 0x3 for addr, data, bus in frames
           if addr == LAT_CANDIDATE_ADDR and bus == LAT_CANDIDATE_BUS and ((data[4] >> 2) & 0x3F) <= 6}
    assert set(low) == {1, 2, 3, 4, 5, 6}, low
    assert set(low.values()) == {6 & 0x3}

    # GW_Syn_All has stepped but OEM hasn't restarted yet: the window keeps OEM's reset.
    cs.secoc_reset = 6
    _, frames = controller.update(cc.as_reader(), cc_sp, cs, 20_000_000)
    flags = {data[4] & 0x3 for addr, data, bus in frames
             if addr == LAT_CANDIDATE_ADDR and bus == LAT_CANDIDATE_BUS and ((data[4] >> 2) & 0x3F) >= 30}
    assert flags == {5 & 0x3}, flags


class TestReleaseAfterTorqueOverride(unittest.TestCase):
  def test_letting_go_does_not_resume_toward_a_far_target(self):
    # Route 000000f6 (2026-11-14, left onto Doane St): the driver held the car straight (torque override) while the
    # command was a leftover left turn; on letting go the controller resumed toward it and steered hard left at 16 mph.
    cp, sp, cs = _make_car()
    controller = CarController(DBC[CAR.FISKER_OCEAN], cp, sp)
    controller.secoc_key_verified = True
    controller.secoc_key = bytes(16)
    cc_sp = structs.CarControlSP()
    cc_sp.mads.active = True
    controller.was_lat_engaged_prev = True
    controller.was_lat_active_prev = True
    controller.was_tx_lateral_prev = True
    controller.engage_delay_remaining = 0
    cs.eps_lat_sts = 2
    cc = structs.CarControl(latActive=True)
    cc.actuators.steeringAngleDeg = 140.0
    cs.out = structs.CarState(steeringAngleDeg=11.0, vEgoRaw=6.9, steeringTorque=1.5)
    cs.driver_intervening = True
    controller._prev_steer_angle = 11.0
    controller.apply_angle_last = 11.0
    controller.update(cc.as_reader(), cc_sp, cs, 0)
    assert controller._angle_override_latch

    cs.driver_intervening = False                   # the driver lets go
    cs.out.steeringTorque = 0.0
    for _ in range(100):
      controller.update(cc.as_reader(), cc_sp, cs, 0)
      assert not controller.was_lat_active_prev
    assert abs(controller.apply_angle_last - 11.0) < 2.0   # the wheel command stays at the wheel

    cc.actuators.steeringAngleDeg = 13.0            # the command comes back near the wheel: steer again
    for _ in range(controller.ENGAGE_DELAY_TICKS + 3):
      controller.update(cc.as_reader(), cc_sp, cs, 0)
    assert controller.was_lat_active_prev

  def test_ordinary_torque_override_still_resumes(self):
    cp, sp, cs = _make_car()
    controller = CarController(DBC[CAR.FISKER_OCEAN], cp, sp)
    controller.secoc_key_verified = True
    controller.secoc_key = bytes(16)
    cc_sp = structs.CarControlSP()
    cc_sp.mads.active = True
    controller.was_lat_engaged_prev = True
    controller.was_lat_active_prev = True
    controller.was_tx_lateral_prev = True
    controller.engage_delay_remaining = 0
    cs.eps_lat_sts = 2
    cc = structs.CarControl(latActive=True)
    cc.actuators.steeringAngleDeg = 2.0
    cs.out = structs.CarState(steeringAngleDeg=8.0, vEgoRaw=25.0, steeringTorque=1.5)
    cs.driver_intervening = True
    controller.update(cc.as_reader(), cc_sp, cs, 0)
    assert not controller._angle_override_latch
    cs.driver_intervening = False
    for _ in range(controller.ENGAGE_DELAY_TICKS + 3):
      controller.update(cc.as_reader(), cc_sp, cs, 0)
    assert controller.was_lat_active_prev
