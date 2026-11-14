#!/usr/bin/env python3
import importlib
import unittest
from unittest import mock

import numpy as np

from opendbc.car.fisker.values import CarControllerParams, FiskerSafetyFlags
from opendbc.car.structs import CarParams
import opendbc.safety.tests.common as common
from opendbc.safety.tests.common import CANPackerSafety
from opendbc.safety.tests.libsafety import libsafety_py

ICC_SETTINGS = 0x52A
ICC_FRAME = bytes.fromhex("BE2026A043008220")
TIMEOUT_US = 500_000

GAS_MSG = 0x214      # VCU_0x214, 16 bytes
ACCEL_MSG = 0x121
LONG_MSGS = (0x121, 0x117, 0x118)
LONG_RELAY_TIMEOUT_US = 60_000
INIT_DONE_US = 1_000_000   # past the relay windows that are open right after a safety mode init

# the messages every Fisker configuration needs to see: (addr, bus)
COMMON_RX_MSGS = ((0x115, 0), (0x318, 0), (0x1C2, 0), (0x1C4, 0), (0x313, 2), (0x514, 0))


class TestFiskerIccSettings(unittest.TestCase):
  """ICC_0x52A: openpilot re-sends the ICC's frame on bus 2; panda drops the ICC's original
  (bus 0 -> 2) only while openpilot's copies keep arriving."""

  def setUp(self):
    self.safety = libsafety_py.libsafety
    self.safety.set_timer(0)  # init_tests() also resets the timer to 0; keep init's timestamp consistent
    self.safety.set_safety_hooks(CarParams.SafetyModel.fisker, 0)
    self.safety.init_tests()

  def _tx(self, bus):
    return self.safety.safety_tx_hook(common.make_msg(bus, ICC_SETTINGS, dat=ICC_FRAME))

  def _fwd(self, bus):
    return self.safety.safety_fwd_hook(bus, ICC_SETTINGS)

  def test_tx_allowed_on_adas_bus_only(self):
    for controls_allowed in (False, True):
      self.safety.set_controls_allowed(controls_allowed)
      self.assertTrue(self._tx(2))
      self.assertFalse(self._tx(0))
      self.assertFalse(self._tx(1))

  def test_tx_allowed_with_long_flag(self):
    self.safety.set_safety_hooks(CarParams.SafetyModel.fisker, 1)
    self.assertTrue(self._tx(2))

  def test_blocked_at_init(self):
    # openpilot is already sending when the safety mode starts: block from the first ICC frame
    self.assertEqual(self._fwd(0), -1)
    self.safety.set_timer(TIMEOUT_US - 1)
    self.assertEqual(self._fwd(0), -1)

  def test_forwarded_if_openpilot_never_sends(self):
    self.safety.set_timer(TIMEOUT_US)
    self.assertEqual(self._fwd(0), 2)

  def test_blocked_while_openpilot_sends(self):
    # each of our frames extends the window past the init window
    for t in range(0, 5_000_000, 200_000):
      self.safety.set_timer(t)
      self.assertTrue(self._tx(2))
      self.assertEqual(self._fwd(0), -1)
    self.safety.set_timer(t + TIMEOUT_US - 1)
    self.assertEqual(self._fwd(0), -1)
    self.safety.set_timer(t + TIMEOUT_US)
    self.assertEqual(self._fwd(0), 2)

  def test_fallback_to_icc_when_openpilot_stops(self):
    self.assertTrue(self._tx(2))
    self.safety.set_timer(TIMEOUT_US)
    self.assertEqual(self._fwd(0), 2)

    # resumes blocking as soon as openpilot sends again
    self.assertTrue(self._tx(2))
    self.assertEqual(self._fwd(0), -1)

  def test_window_restarts_on_safety_mode_init(self):
    self.safety.set_timer(2 * TIMEOUT_US)
    self.assertEqual(self._fwd(0), 2)
    self.safety.set_safety_hooks(CarParams.SafetyModel.fisker, 0)
    self.assertEqual(self._fwd(0), -1)

  def test_adas_side_not_blocked(self):
    # only the ICC -> ADAS direction is intercepted
    self.assertTrue(self._tx(2))
    self.assertEqual(self._fwd(2), 0)

  def test_no_relay_malfunction(self):
    # 0x52A on bus 2 must not trip relay malfunction (that would stop all forwarding)
    self.safety.safety_rx_hook(common.make_msg(2, ICC_SETTINGS, dat=ICC_FRAME))
    self.assertFalse(self.safety.get_relay_malfunction())


class TestFiskerIcc0x35B(unittest.TestCase):
  """ICC_0x35B: optional relay — the ICC's original is only blocked once openpilot's copies are
  actually arriving on bus 2, and forwarding resumes if they stop."""

  ADDR = 0x35B

  def setUp(self):
    self.safety = libsafety_py.libsafety
    self.safety.set_timer(0)  # init_tests() also resets the timer to 0; keep init's timestamp consistent
    self.safety.set_safety_hooks(CarParams.SafetyModel.fisker, 0)
    self.safety.init_tests()

  def _tx(self, bus):
    return self.safety.safety_tx_hook(common.make_msg(bus, self.ADDR, 8))

  def _fwd(self, bus):
    return self.safety.safety_fwd_hook(bus, self.ADDR)

  def test_tx_allowed_on_adas_bus_only(self):
    for param in (0, 1):
      self.safety.set_safety_hooks(CarParams.SafetyModel.fisker, param)
      self.assertTrue(self._tx(2))
      self.assertFalse(self._tx(0))
      self.assertFalse(self._tx(1))

  def test_forwarded_while_openpilot_not_relaying(self):
    # no overrides configured -> openpilot never sends 0x35B -> ICC's frame always forwarded
    for t in (0, TIMEOUT_US, 10 * TIMEOUT_US):
      self.safety.set_timer(t)
      self.assertEqual(self._fwd(0), 2)

  def test_blocked_while_openpilot_relays(self):
    for t in range(0, 5_000_000, 200_000):
      self.safety.set_timer(t)
      self.assertTrue(self._tx(2))
      self.assertEqual(self._fwd(0), -1)
    self.safety.set_timer(t + TIMEOUT_US - 1)
    self.assertEqual(self._fwd(0), -1)
    self.safety.set_timer(t + TIMEOUT_US)
    self.assertEqual(self._fwd(0), 2)

  def test_reset_on_safety_mode_init(self):
    self.assertTrue(self._tx(2))
    self.assertEqual(self._fwd(0), -1)
    self.safety.set_safety_hooks(CarParams.SafetyModel.fisker, 0)
    self.assertEqual(self._fwd(0), 2)

  def test_independent_of_0x52a(self):
    # relaying 0x52A doesn't block 0x35B, and vice versa
    self.assertTrue(self.safety.safety_tx_hook(common.make_msg(2, ICC_SETTINGS, dat=ICC_FRAME)))
    self.assertEqual(self._fwd(0), 2)
    self.safety.set_timer(TIMEOUT_US)
    self.assertTrue(self._tx(2))
    self.assertEqual(self._fwd(0), -1)
    self.assertEqual(self.safety.safety_fwd_hook(0, ICC_SETTINGS), 2)

  def test_adas_side_not_blocked(self):
    self.assertTrue(self._tx(2))
    self.assertEqual(self._fwd(2), 0)


class TestFiskerCruiseState(unittest.TestCase):
  """Cruise engagement follows the ADAS module's ACC state (ADAS_0x313, bus 2), and with the
  FiskerBasicCruise SP param also the VCU's basic cruise (VCU_0x358, bus 0)."""

  # ADAS_Sts_ACC_ICC: Active, Override, Standstill_active, Standstill_wait, Standstill_GoNotification
  ENGAGED_STATES = {3, 4, 5, 6, 11}
  FISKER_PARAM_SP_VCU_BASIC_CRUISE = 1

  def setUp(self):
    self.safety = libsafety_py.libsafety
    self._init(vcu_basic_cruise=False)

  def tearDown(self):
    self.safety.set_current_safety_param_sp(0)   # libsafety is shared: leave the other tests the default

  def _init(self, vcu_basic_cruise):
    self.safety.set_current_safety_param_sp(self.FISKER_PARAM_SP_VCU_BASIC_CRUISE if vcu_basic_cruise else 0)
    self.safety.set_safety_hooks(CarParams.SafetyModel.fisker, 0)
    self.safety.init_tests()

  def _state_msg(self, addr, bus, acc_state):
    dat = bytearray(8)
    dat[4] = acc_state & 0x0F  # 4-bit @ Motorola start 35 -> data[4] low nibble
    return common.make_msg(bus, addr, dat=bytes(dat))

  def _rx_acc_state(self, acc_state, addr=0x313, bus=2):
    self.safety.safety_rx_hook(self._state_msg(addr, bus, acc_state))

  def test_engaged_states(self):
    for acc_state in range(16):
      with self.subTest(acc_state=acc_state):
        self._rx_acc_state(0)
        self.assertFalse(self.safety.get_controls_allowed())
        self._rx_acc_state(acc_state)
        self.assertEqual(self.safety.get_controls_allowed(), acc_state in self.ENGAGED_STATES)

  def test_disengage_on_acc_exit(self):
    for acc_state in set(range(16)) - self.ENGAGED_STATES:
      with self.subTest(acc_state=acc_state):
        self._rx_acc_state(0)
        self._rx_acc_state(3)
        self.assertTrue(self.safety.get_controls_allowed())
        self._rx_acc_state(acc_state)
        self.assertFalse(self.safety.get_controls_allowed())

  def test_stays_engaged_through_standstill(self):
    self._rx_acc_state(0)
    for acc_state in (3, 5, 6, 11, 3, 4):
      self._rx_acc_state(acc_state)
      self.assertTrue(self.safety.get_controls_allowed())

  def test_other_sources_ignored(self):
    # 0x313 only counts from the ADAS side (bus 2), 0x358 only from the vehicle side (bus 0)
    for addr, bus in ((0x358, 2), (0x313, 0)):
      with self.subTest(addr=hex(addr), bus=bus):
        self._rx_acc_state(0)
        self._rx_acc_state(3, addr=addr, bus=bus)
        self.assertFalse(self.safety.get_controls_allowed())

  def test_vcu_basic_cruise_ignored_by_default(self):
    # with ACC coded the VCU's basic cruise means nothing (0x358 isn't even an rx check)
    for vcu_state in (3, 4):
      with self.subTest(vcu_state=vcu_state):
        self._rx_acc_state(0)
        self._rx_acc_state(vcu_state, addr=0x358, bus=0)
        self.assertFalse(self.safety.get_controls_allowed())

  def test_vcu_basic_cruise_engages(self):
    # Cars without ACC coded (FiskerBasicCruise): ADAS stays Off, VCU basic cruise (0x358, bus 0) engages
    self._init(vcu_basic_cruise=True)
    for vcu_state in range(16):
      with self.subTest(vcu_state=vcu_state):
        self._rx_acc_state(0)
        self._rx_acc_state(0, addr=0x358, bus=0)
        self.assertFalse(self.safety.get_controls_allowed())
        self._rx_acc_state(vcu_state, addr=0x358, bus=0)
        self.assertEqual(self.safety.get_controls_allowed(), vcu_state in (3, 4))

  def test_acc_main_never_set(self):
    # MADS falling-edges on acc_main_on, so neither cruise source sets it (see fisker_rx_hook)
    self._init(vcu_basic_cruise=True)
    for addr, bus in ((0x313, 2), (0x358, 0)):
      for state in range(16):
        with self.subTest(addr=hex(addr), state=state):
          self._rx_acc_state(state, addr=addr, bus=bus)
          self.assertFalse(self.safety.get_acc_main_on())

  def test_adas_off_does_not_drop_vcu_cruise(self):
    self._init(vcu_basic_cruise=True)
    self._rx_acc_state(3, addr=0x358, bus=0)
    self.assertTrue(self.safety.get_controls_allowed())
    self._rx_acc_state(0)  # ADAS ACC (not coded) keeps reporting Off
    self.assertTrue(self.safety.get_controls_allowed())
    self._rx_acc_state(2, addr=0x358, bus=0)  # brake -> Standby
    self.assertFalse(self.safety.get_controls_allowed())


class TestFiskerSteerBaselineResync(unittest.TestCase):
  """While not injecting (OEM 0x1D0 passing through), the angle rate-limiter baseline
  (desired_angle_last) must track the measured wheel angle. Cruise stays on through the
  not-steering window, so the generic not-controls-allowed reset never fires; without the
  fisker-specific resync, a re-engage after the wheel moved crawls from a stale baseline and
  rejects frames until it catches up, and the EPS aborts (observed on-vehicle 2026-10-25)."""

  TX_MSGS = None
  ANGLE_ZERO = 12480

  def setUp(self):
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.fisker, 0)
    self.safety.init_tests()

  def _meas(self, deg):
    raw = int(deg * 16) + self.ANGLE_ZERO
    dat = bytearray(8)
    dat[2] = (raw >> 8) & 0xFF
    dat[3] = raw & 0xFF
    self.safety.safety_rx_hook(common.make_msg(0, 0x1C2, dat=bytes(dat)))

  def test_baseline_tracks_measured_while_not_armed(self):
    # Engage cruise so controls ARE allowed (the generic reset would not fire here).
    dat = bytearray(8)
    dat[4] = 3
    self.safety.safety_rx_hook(common.make_msg(2, 0x313, dat=bytes(dat)))
    self.assertTrue(self.safety.get_controls_allowed())
    # Advance past the candidate-arm timeout so we count as "not injecting" (no 0x5D0 recently).
    self.safety.set_timer(1_000_000)
    # Driver turns the wheel to +40 deg while openpilot is not steering (no candidates).
    for _ in range(5):
      self._meas(40.0)
    # Baseline now sits at the measured angle, not a stale 0, so a re-engage near +40 passes.
    self.assertEqual(self.safety.get_desired_angle_last(), int(40.0 * 16))


class TestFiskerCanIgnition(unittest.TestCase):
  """The panda's voltage ignition line is never used on the Ocean (it reads high permanently),
  so CAN ignition from BCM_0x333.BCM_PwrMod (bus 0) is the only thing that starts openpilot."""

  PWR_MOD_RUN = 4

  def setUp(self):
    self.safety = libsafety_py.libsafety
    self.safety.init_tests()

  def _msg(self, pwr_mod, bus=0, other_bits=0x00, length=8):
    dat = bytearray(length)
    dat[2] = other_bits | ((pwr_mod & 0x7) << 3)  # 3-bit @ Motorola start 21 -> data[2] bits 5..3
    return common.make_msg(bus, 0x333, dat=bytes(dat))

  def test_ignition_on(self):
    for pwr_mod in range(8):
      with self.subTest(pwr_mod=pwr_mod):
        self.safety.set_ignition_can(False)
        self.safety.ignition_can_hook(self._msg(pwr_mod))
        self.assertEqual(self.safety.get_ignition_can(), pwr_mod == self.PWR_MOD_RUN)

  def test_ignition_off(self):
    for pwr_mod in set(range(8)) - {self.PWR_MOD_RUN}:
      with self.subTest(pwr_mod=pwr_mod):
        self.safety.ignition_can_hook(self._msg(self.PWR_MOD_RUN))
        self.assertTrue(self.safety.get_ignition_can())
        self.safety.ignition_can_hook(self._msg(pwr_mod))
        self.assertFalse(self.safety.get_ignition_can())

  def test_neighboring_signals_ignored(self):
    # BCM_PwrModVld (data[2] bits 7..6) and BCM_IG1Sts (bits 2..1) share the byte
    self.safety.ignition_can_hook(self._msg(self.PWR_MOD_RUN, other_bits=0xC7))
    self.assertTrue(self.safety.get_ignition_can())
    self.safety.ignition_can_hook(self._msg(3, other_bits=0xC7))
    self.assertFalse(self.safety.get_ignition_can())

  def test_other_buses_and_lengths_ignored(self):
    for bus, length in ((1, 8), (2, 8), (0, 16)):
      with self.subTest(bus=bus, length=length):
        self.safety.set_ignition_can(False)
        self.safety.ignition_can_hook(self._msg(self.PWR_MOD_RUN, bus=bus, length=length))
        self.assertFalse(self.safety.get_ignition_can())


class TestFiskerHooks(unittest.TestCase):
  """Signal decoding in the rx hook, actuator checks in the tx hook, and the forwarding blocks."""

  ANGLE_ZERO = 12480
  ACCEL_INACTIVE = 32773
  ACCEL_MAX = 34822   # +1.0 m/s^2
  LAT_RELAY_TIMEOUT_US = 60_000

  def setUp(self):
    self.safety = libsafety_py.libsafety
    self._init(0)

  def _init(self, param):
    self.safety.set_timer(0)
    self.safety.set_safety_hooks(CarParams.SafetyModel.fisker, param)
    self.safety.init_tests()

  def _rx(self, addr, dat, bus=0):
    self.safety.safety_rx_hook(common.make_msg(bus, addr, dat=bytes(dat)))

  def _tx(self, addr, dat, bus=0):
    return self.safety.safety_tx_hook(common.make_msg(bus, addr, dat=bytes(dat)))

  def _u16_msg(self, raw):
    # angle (0x1D0) and accel (0x121) are both big-endian 16-bit in bytes 2,3
    dat = bytearray(8)
    dat[2], dat[3] = (raw >> 8) & 0xFF, raw & 0xFF
    return dat

  def test_driver_torque(self):
    for torque in (0, 100, 0xFFF):
      with self.subTest(torque=torque):
        dat = bytearray(8)
        dat[2], dat[3] = torque >> 4, (torque & 0xF) << 4
        self.safety.set_torque_driver(0, 0)
        self._rx(0x1C4, dat)
        self.assertEqual(self.safety.get_torque_driver_max(), torque)

  def test_vehicle_speed_and_brake(self):
    for brake in (False, True):
      with self.subTest(brake=brake):
        dat = bytearray(8)
        dat[1] = 0x40 if brake else 0xBF  # ESP_BrkPedlSts is bit 6; the others mustn't matter
        dat[5], dat[6] = 0x03, 0xE8       # 1000 * 0.1 km/h = 100 km/h
        self._rx(0x318, dat)
        self.assertEqual(self.safety.get_brake_pressed_prev(), brake)
        self.assertAlmostEqual(self.safety.get_vehicle_speed_max(), 100 / 3.6, places=3)

  def test_vehicle_moving(self):
    for whl, moving in ((0, False), (1, True), (0x3FFF, True)):
      with self.subTest(whl=whl):
        dat = bytearray(8)
        dat[1], dat[2] = 0xC0 | (whl >> 8), whl & 0xFF  # top two bits of byte 1 aren't the wheel speed
        self._rx(0x115, dat)
        self.assertEqual(self.safety.get_vehicle_moving(), moving)

  def test_mads_button_long_press_only(self):
    for value in range(4):
      with self.subTest(value=value):
        dat = bytearray(8)
        dat[4] = 0xFC | value
        self._rx(0x514, dat)
        self.assertEqual(self.safety.get_mads_button_press(), int(value == 2))

  def test_steer_angle(self):
    self.safety.set_controls_allowed(True)
    self.safety.set_desired_angle_last(0)
    self.assertTrue(self._tx(0x1D0, self._u16_msg(self.ANGLE_ZERO)))
    # a 90 deg jump in one frame is far past the rate limit
    self.assertFalse(self._tx(0x1D0, self._u16_msg(self.ANGLE_ZERO + 90 * 16)))

  def test_lateral_activation(self):
    for allowed in (False, True):
      for lat_req in range(4):
        with self.subTest(allowed=allowed, lat_req=lat_req):
          self.safety.set_controls_allowed(allowed)
          dat = bytearray(8)
          dat[3] = lat_req << 6
          self.assertEqual(self._tx(0x1C0, dat), allowed or lat_req == 0)

  def test_oem_lateral_activation_blocked_while_sending(self):
    self.safety.set_timer(1_000_000)
    self.assertEqual(self.safety.safety_fwd_hook(2, 0x1C0), 0)  # idle until openpilot sends
    self.assertTrue(self._tx(0x1C0, bytearray(8)))
    self.assertEqual(self.safety.safety_fwd_hook(2, 0x1C0), -1)
    self.safety.set_timer(1_000_000 + self.LAT_RELAY_TIMEOUT_US - 1)
    self.assertEqual(self.safety.safety_fwd_hook(2, 0x1C0), -1)
    self.safety.set_timer(1_000_000 + self.LAT_RELAY_TIMEOUT_US)
    self.assertEqual(self.safety.safety_fwd_hook(2, 0x1C0), 0)

  def test_accel(self):
    self.safety.set_controls_allowed(True)
    self.assertFalse(self._tx(0x121, self._u16_msg(self.ACCEL_INACTIVE)))  # only in the TX list with long
    self._init(1)
    self.safety.set_controls_allowed(True)
    self.assertTrue(self._tx(0x121, self._u16_msg(self.ACCEL_INACTIVE)))
    self.assertTrue(self._tx(0x121, self._u16_msg(self.ACCEL_MAX)))
    self.assertFalse(self._tx(0x121, self._u16_msg(self.ACCEL_MAX + 1)))
    self.safety.set_controls_allowed(False)
    self.assertTrue(self._tx(0x121, self._u16_msg(self.ACCEL_INACTIVE)))
    self.assertFalse(self._tx(0x121, self._u16_msg(self.ACCEL_MAX)))

  def test_injected_frames_kept_off_adas_bus(self):
    for long in (0, 1):
      self._init(long)
      for addr in (0x1D0, 0x1C0, 0x121, 0x117, 0x118):
        with self.subTest(long=long, addr=hex(addr)):
          blocked = addr in (0x1D0, 0x1C0) or bool(long)
          self.assertEqual(self.safety.safety_fwd_hook(0, addr), -1 if blocked else 2)

  LONG_RELAY_TIMEOUT_US = 60_000

  def test_oem_long_frames_flow_until_we_send_our_own(self):
    for long in (0, 1):
      for allowed in (False, True):
        with self.subTest(long=long, allowed=allowed):
          self._init(long)
          self.safety.set_controls_allowed(allowed)
          for addr in (0x121, 0x117, 0x118):
            self.assertEqual(self.safety.safety_fwd_hook(2, addr), 0)

  def test_oem_long_frame_blocked_while_ours_is_sent(self):
    # One sender per message: ours while we send it, the OEM's otherwise, whatever controls_allowed says (with the
    # old controls_allowed rule both 0x121 streams reached the VCU whenever we sent with controls not allowed).
    for allowed in (False, True):
      for addr in (0x121, 0x117, 0x118):
        with self.subTest(allowed=allowed, addr=hex(addr)):
          self._init(1)
          self.safety.set_controls_allowed(allowed)
          t = 1_000_000
          self.safety.set_timer(t)
          dat = self._u16_msg(self.ACCEL_INACTIVE) if addr == 0x121 else bytearray(8)
          self.assertTrue(self._tx(addr, dat))
          self.assertEqual(self.safety.safety_fwd_hook(2, addr), -1)
          for other in {0x121, 0x117, 0x118} - {addr}:           # per message: the others still flow
            self.assertEqual(self.safety.safety_fwd_hook(2, other), 0)
          self.safety.set_timer(t + self.LONG_RELAY_TIMEOUT_US - 1)
          self.assertEqual(self.safety.safety_fwd_hook(2, addr), -1)
          self.safety.set_timer(t + self.LONG_RELAY_TIMEOUT_US)   # we stopped: the OEM's resumes
          self.assertEqual(self.safety.safety_fwd_hook(2, addr), 0)

  def test_rejected_accel_does_not_block_oem_accel(self):
    self._init(1)
    self.safety.set_controls_allowed(True)
    self.safety.set_timer(1_000_000)
    self.assertFalse(self._tx(0x121, self._u16_msg(self.ACCEL_MAX + 1)))
    self.assertEqual(self.safety.safety_fwd_hook(2, 0x121), 0)

  def test_long_status_frames_cannot_request_gear_epb_or_braking(self):
    from opendbc.can import CANPacker
    from opendbc.car import Bus
    from opendbc.car.fisker.fiskercan import FiskerCAN
    from opendbc.car.fisker.values import CAR, DBC
    fc = FiskerCAN(None, CANPacker(DBC[CAR.FISKER_OCEAN][Bus.pt]))
    # (byte, mask) of each request that must stay 0: 0x117 park gear, emergency EPB, emergency/park standstill;
    # 0x118 HBA, ESP standstill, brake prefill, AEB activation
    forbidden = {0x117: ((4, 0x01), (4, 0x08), (5, 0x10), (3, 0x10), (3, 0x20)),
                 0x118: ((3, 0x01), (3, 0x04), (3, 0x20), (4, 0x10), (4, 0x20), (5, 0x01), (5, 0x08))}
    self._init(1)
    for allowed in (False, True):
      self.safety.set_controls_allowed(allowed)
      for long_active in (False, True):
        for gas in (False, True):
          frames = {0x117: fc.create_long_status(long_active, gas, gear_req=0, counter=3)[1],
                    0x118: fc.create_long_esp_handshake(long_active, gas, counter=3)[1]}
          for addr, dat in frames.items():
            with self.subTest(addr=hex(addr), allowed=allowed, long_active=long_active, gas=gas):
              self.assertTrue(self._tx(addr, dat))                # what openpilot sends goes out
              for byte, mask in forbidden[addr]:
                bad = bytearray(dat)
                bad[byte] |= mask
                self.assertFalse(self._tx(addr, bad), f"byte {byte} mask {mask:#x}")

  def test_long_relay_off_without_long_flag(self):
    self._init(0)
    self.safety.set_timer(1_000_000)
    for addr in (0x121, 0x117, 0x118):
      self.assertFalse(self._tx(addr, bytearray(8)))           # not in the TX list
      self.assertEqual(self.safety.safety_fwd_hook(2, addr), 0)


class TestFiskerSafetyBase(common.CarSafetyTest, common.AngleSteeringSafetyTest):
  SAFETY_PARAM = 0

  TX_MSGS = [[0x1D0, 0], [0x5D0, 1], [0x1C0, 0], [0x52A, 2], [0x35B, 2]]
  RELAY_MALFUNCTION_ADDRS = {0: (0x1D0, 0x1C0)}
  # Forwarding is time-relay based, not static. This is the picture while openpilot sends nothing, which is what
  # the standard test_fwd_hook sees: only the frames openpilot injects on bus 0 are kept off the ADAS's bus. The
  # relays themselves (what gets blocked while openpilot sends) are tested in test_fisker.py.
  FWD_BLACKLISTED_ADDRS = {0: [0x1D0, 0x1C0]}

  STANDSTILL_THRESHOLD = 0
  GAS_PRESSED_THRESHOLD = 1   # percent pedal, as carstate: pressed above 1 %

  # Angle command limits
  STEER_ANGLE_MAX = CarControllerParams.ANGLE_LIMITS.STEER_ANGLE_MAX
  STEER_ANGLE_TEST_MAX = 700  # the signal's range is +-780 deg
  DEG_TO_CAN = 16
  ANGLE_RATE_BP = CarControllerParams.ANGLE_LIMITS.ANGLE_RATE_LIMIT_UP[0]
  ANGLE_RATE_UP = CarControllerParams.ANGLE_LIMITS.ANGLE_RATE_LIMIT_UP[1]
  ANGLE_RATE_DOWN = CarControllerParams.ANGLE_LIMITS.ANGLE_RATE_LIMIT_DOWN[1]

  def setUp(self):
    self.packer = CANPackerSafety("fisker_ocean_adas")
    self.safety = libsafety_py.libsafety
    self.safety.set_timer(0)  # init_tests() also resets the timer to 0; keep init's timestamp consistent
    self.safety.set_current_safety_param_sp(0)
    self.safety.set_safety_hooks(CarParams.SafetyModel.fisker, self.SAFETY_PARAM)
    self.safety.init_tests()
    self.safety.set_timer(INIT_DONE_US)
    self.speed = 0.
    self.brake = False

  def _angle_cmd_msg(self, angle: float, enabled: bool, increment_timer: bool = True):
    # 0x1D0 has no enable bit: steering counts as requested whenever the frame is sent
    values = {"ADAS_LatCtrl_SteerAnReq": angle}
    return self.packer.make_can_msg_safety("ADAS_0x1D0", 0, values)

  def _angle_meas_msg(self, angle: float):
    values = {"EPS_SteerWhlAgSig": angle}
    return self.packer.make_can_msg_safety("EPS_0x1C2", 0, values)

  def _esp_msg(self):
    # ESP_0x318 carries both the vehicle speed and the brake pedal, so each helper keeps the other's last value
    values = {"ESP_VehSpd": self.speed * 3.6, "ESP_BrkPedlSts": 1 if self.brake else 0}
    return self.packer.make_can_msg_safety("ESP_0x318", 0, values)

  def _speed_msg(self, speed):
    self.speed = speed
    return self._esp_msg()

  def _speed_msg_2(self, speed):
    return None

  def _vehicle_moving_msg(self, speed: float):
    # NOTE: built from the bytes fisker_rx_hook reads (byte 1 low 6 bits + byte 2), like test_fisker.py's
    # test_vehicle_moving. The DBC puts ESP_WhlSpd_RF (21|14@0+) one byte later, in bytes 2-3; a frame packed
    # from the DBC does not drive vehicle_moving the way these tests need.
    whl = round(speed * 3.6 / 0.0625)
    dat = bytearray(8)
    dat[1], dat[2] = whl >> 8, whl & 0xFF
    return common.make_msg(0, 0x115, dat=bytes(dat))

  def _user_brake_msg(self, brake):
    self.brake = bool(brake)
    return self._esp_msg()

  def _user_gas_msg(self, gas):
    values = {"VCU_APSPerc": gas}
    return self.packer.make_can_msg_safety("VCU_0x214", 0, values)

  def _pcm_status_msg(self, enable):
    values = {"ADAS_Sts_ACC_ICC": 3 if enable else 2}  # Active / Standby
    return self.packer.make_can_msg_safety("ADAS_0x313", 2, values)

  def _lkas_button_msg(self, enabled):
    values = {"MFS_RiBtnSouth": 2 if enabled else 0}   # the long press is the MADS button
    return self.packer.make_can_msg_safety("MFS_0x514", 0, values)

  def _rx_common_msgs(self):
    for addr, bus in COMMON_RX_MSGS:
      self.assertTrue(self._rx(common.make_msg(bus, addr, 8)))

  def test_tx_hook_on_wrong_safety_mode(self):
    # The two Fisker configurations share their TX messages, like the other brands' variants common.py exempts.
    # (Patched on the module common.py imports, which is not this one when the test runner loads it by path.)
    module = importlib.import_module("opendbc.safety.tests.test_fisker")
    for name in ("TestFiskerStockSafety", "TestFiskerLongSafety"):
      self.assertTrue(issubclass(getattr(module, name), common.SafetyTest))
    other = "TestFiskerLongSafety" if type(self).__name__ == "TestFiskerStockSafety" else "TestFiskerStockSafety"
    with mock.patch.object(getattr(module, other), "TX_MSGS", None):
      super().test_tx_hook_on_wrong_safety_mode()

  def test_angle_cmd_when_disabled(self):
    # 0x1D0 has no enable bit, so unlike the standard test there is no "not requested" frame while controls are
    # allowed. With controls allowed the command is held to the rate limits (test_angle_cmd_when_enabled); with
    # controls not allowed only the measured angle may be sent.
    for controls_allowed in (True, False):
      self.safety.set_controls_allowed(controls_allowed)
      for angle_meas in np.arange(-90, 91, 10):
        self._reset_angle_measurement(angle_meas)
        for angle_cmd in np.arange(-90, 91, 10):
          self._set_prev_desired_angle(angle_cmd)
          should_tx = controls_allowed or angle_cmd == angle_meas
          self.assertEqual(should_tx, self._tx(self._angle_cmd_msg(angle_cmd, controls_allowed)))

  def test_gas_does_not_change_controls_allowed(self):
    for controls_allowed in (False, True):
      for lateral in (False, True):
        with self.subTest(controls_allowed=controls_allowed, lateral=lateral):
          self._rx(self._user_gas_msg(0))
          self.safety.set_controls_allowed(controls_allowed)
          self.safety.set_controls_allowed_lateral(lateral)
          for gas in (50, 50, 0, 0):
            self._rx(self._user_gas_msg(gas))
            self.assertEqual(self.safety.get_controls_allowed(), controls_allowed)
            self.assertEqual(self.safety.get_controls_allowed_lateral(), lateral)

  def test_steering_allowed_with_gas_pressed(self):
    self._rx(self._user_gas_msg(50))
    self._rx(self._user_gas_msg(50))
    self._reset_angle_measurement(0)
    self.safety.set_controls_allowed(True)
    self._set_prev_desired_angle(0)
    self.assertTrue(self._tx(self._angle_cmd_msg(1, True)))
    self.assertTrue(self._tx(self.packer.make_can_msg_safety("ADAS_0x1C0", 0, {"ADAS_LatCtrl_Req": 1})))


class TestFiskerStockSafety(TestFiskerSafetyBase, common.LongitudinalAccelSafetyTest):
  """Steering only: what runs on the car."""

  LONGITUDINAL = False
  MAX_ACCEL = CarControllerParams.ACCEL_MAX
  MIN_ACCEL = CarControllerParams.ACCEL_MIN

  def _accel_msg(self, accel: float):
    values = {"ADAS_LgtCtrl_AccelReq": accel}
    return self.packer.make_can_msg_safety("ADAS_0x121", 0, values)

  # The accelerator pedal is only read with openpilot longitudinal. Without it VCU_0x214 is not an rx check, so
  # the pedal never reaches the rx hook and the message is not a liveness requirement either.
  def test_prev_gas(self):
    for gas in (self.GAS_PRESSED_THRESHOLD + 1, 100, 0):
      self._rx(self._user_gas_msg(gas))
      self.assertFalse(self.safety.get_gas_pressed_prev())

  def test_no_disengage_on_gas(self):
    self._rx(self._user_gas_msg(0))
    self.safety.set_controls_allowed(True)
    self._rx(self._user_gas_msg(self.GAS_PRESSED_THRESHOLD + 1))
    self.assertTrue(self.safety.get_controls_allowed())
    self.assertTrue(self.safety.get_longitudinal_allowed())

  def test_gas_message_not_required(self):
    self._rx_common_msgs()
    self.safety.safety_tick_current_safety_config()
    self.assertTrue(self.safety.safety_config_valid())

  def test_long_frames_not_sent(self):
    for controls_allowed in (False, True):
      self.safety.set_controls_allowed(controls_allowed)
      for addr in LONG_MSGS:
        self.assertFalse(self._tx(common.make_msg(0, addr, 8)))
      self.assertFalse(self._tx(self._accel_msg(0)))


class TestFiskerLongSafety(TestFiskerSafetyBase, common.LongitudinalAccelSafetyTest):
  """openpilot longitudinal (debug builds only)."""

  SAFETY_PARAM = FiskerSafetyFlags.LONG_CONTROL

  TX_MSGS = [[0x1D0, 0], [0x5D0, 1], [0x1C0, 0], [0x121, 0], [0x117, 0], [0x118, 0], [0x52A, 2], [0x35B, 2]]
  RELAY_MALFUNCTION_ADDRS = {0: (0x1D0, 0x1C0, 0x121, 0x117, 0x118)}
  FWD_BLACKLISTED_ADDRS = {0: [0x1D0, 0x1C0, 0x121, 0x117, 0x118]}

  MAX_ACCEL = CarControllerParams.ACCEL_MAX
  MIN_ACCEL = CarControllerParams.ACCEL_MIN

  def _accel_msg(self, accel: float):
    values = {"ADAS_LgtCtrl_AccelReq": accel}
    return self.packer.make_can_msg_safety("ADAS_0x121", 0, values)

  def _gas_raw_msg(self, raw, bus=0, length=16):
    dat = bytearray(length)
    dat[6] = raw  # VCU_APSPerc, 0.4 %/LSB
    return common.make_msg(bus, GAS_MSG, dat=bytes(dat))

  def test_gas_threshold(self):
    # carstate: gasPressed = VCU_APSPerc > 1.0 %, which is raw 3 and up
    for raw in (0, 1, 2, 3, 4, 125, 250, 255):
      with self.subTest(raw=raw):
        self.assertTrue(self._rx(self._gas_raw_msg(raw)))
        self.assertEqual(self.safety.get_gas_pressed_prev(), raw > 2)

  def test_gas_other_bytes_ignored(self):
    dat = bytearray(b"\xff" * 16)
    dat[6] = 0
    self._rx(common.make_msg(0, GAS_MSG, dat=bytes(dat)))
    self.assertFalse(self.safety.get_gas_pressed_prev())
    self._rx(self._gas_raw_msg(3))
    self.assertTrue(self.safety.get_gas_pressed_prev())

  def test_gas_only_from_vehicle_bus_and_full_length(self):
    for bus, length in ((1, 16), (2, 16), (0, 8)):
      with self.subTest(bus=bus, length=length):
        self._rx(self._gas_raw_msg(250, bus=bus, length=length))
        self.assertFalse(self.safety.get_gas_pressed_prev())
        self.assertTrue(self.safety.get_longitudinal_allowed() or not self.safety.get_controls_allowed())

  def test_gas_reset_on_safety_init(self):
    self._rx(self._user_gas_msg(50))
    self.assertTrue(self.safety.get_gas_pressed_prev())
    self._reset_safety_hooks()
    self.assertFalse(self.safety.get_gas_pressed_prev())

  def test_accel_rejected_while_gas_pressed(self):
    active_accels = (self.MIN_ACCEL, -1.0, 0.5, self.MAX_ACCEL)
    self._rx(self._user_gas_msg(0))
    self.safety.set_controls_allowed(True)
    for accel in active_accels:
      self.assertTrue(self._tx(self._accel_msg(accel)))

    self._rx(self._user_gas_msg(50))
    self.assertTrue(self.safety.get_controls_allowed())
    self.assertFalse(self.safety.get_longitudinal_allowed())
    for accel in active_accels:
      self.assertFalse(self._tx(self._accel_msg(accel)), accel)
    self.assertTrue(self._tx(self._accel_msg(self.INACTIVE_ACCEL)))   # the inactive accel always passes
    self.assertTrue(self.safety.get_controls_allowed())

    self._rx(self._user_gas_msg(0))
    self.assertTrue(self.safety.get_longitudinal_allowed())
    for accel in active_accels:
      self.assertTrue(self._tx(self._accel_msg(accel)), accel)

  def test_accel_sweep_while_gas_pressed(self):
    self._rx(self._user_gas_msg(50))
    for controls_allowed in (False, True):
      self.safety.set_controls_allowed(controls_allowed)
      for accel in np.concatenate((np.arange(self.MIN_ACCEL - 1, self.MAX_ACCEL + 1, 0.05), [0, self.INACTIVE_ACCEL])):
        accel = round(accel, 2)
        self.assertEqual(self._tx(self._accel_msg(accel)), accel == self.INACTIVE_ACCEL, accel)

  def test_long_status_frames_pass_while_gas_pressed(self):
    # only the accel is held back: the status / handshake frames carry the override state to the car
    self._rx(self._user_gas_msg(50))
    self.safety.set_controls_allowed(True)
    for addr in (0x117, 0x118):
      self.assertTrue(self._tx(common.make_msg(0, addr, 8)))

  def test_accel_rejected_on_gas_does_not_block_oem_accel(self):
    # one sender per message: a 0x121 the gas check rejects must not also silence the OEM's
    self.safety.set_controls_allowed(True)
    self.assertTrue(self._tx(self._accel_msg(0.5)))
    self.assertEqual(self.safety.safety_fwd_hook(2, ACCEL_MSG), -1)

    self._rx(self._user_gas_msg(50))
    self.safety.set_timer(INIT_DONE_US + LONG_RELAY_TIMEOUT_US)
    self.assertFalse(self._tx(self._accel_msg(0.5)))
    self.assertEqual(self.safety.safety_fwd_hook(2, ACCEL_MSG), 0)
    self.assertTrue(self._tx(self._accel_msg(self.INACTIVE_ACCEL)))
    self.assertEqual(self.safety.safety_fwd_hook(2, ACCEL_MSG), -1)

  def test_brake_still_disengages_with_gas_pressed(self):
    self._rx(self._vehicle_moving_msg(10))
    self._rx(self._user_brake_msg(False))
    self.safety.set_controls_allowed(True)
    self._rx(self._user_gas_msg(50))
    self.assertTrue(self.safety.get_controls_allowed())
    self._rx(self._user_brake_msg(True))
    self.assertFalse(self.safety.get_controls_allowed())
    self.assertFalse(self._tx(self._accel_msg(0.5)))
    self._rx(self._user_gas_msg(0))
    self.assertFalse(self._tx(self._accel_msg(0.5)))
    self.assertTrue(self._tx(self._accel_msg(self.INACTIVE_ACCEL)))

  def test_gas_message_required(self):
    # with openpilot longitudinal the pedal message has to be there, like the other rx checks
    self._rx_common_msgs()
    self.safety.safety_tick_current_safety_config()
    self.assertFalse(self.safety.safety_config_valid())
    self.assertTrue(self._rx(self._gas_raw_msg(0)))
    self.safety.safety_tick_current_safety_config()
    self.assertTrue(self.safety.safety_config_valid())

    # and it has to keep coming
    self.safety.set_controls_allowed(True)
    self.safety.set_timer(INIT_DONE_US + 2_000_000)
    self._rx_common_msgs()
    self.safety.safety_tick_current_safety_config()
    self.assertFalse(self.safety.safety_config_valid())
    self.assertFalse(self.safety.get_controls_allowed())


class TestFiskerVcuBasicCruiseGas(unittest.TestCase):
  """The pedal is read the same way on cars that engage from the VCU's basic cruise (FiskerBasicCruise)."""

  FISKER_PARAM_SP_VCU_BASIC_CRUISE = 1

  def setUp(self):
    self.safety = libsafety_py.libsafety

  def tearDown(self):
    self.safety.set_current_safety_param_sp(0)   # libsafety is shared: leave the other tests the default

  def _init(self, param):
    self.safety.set_current_safety_param_sp(self.FISKER_PARAM_SP_VCU_BASIC_CRUISE)
    self.safety.set_safety_hooks(CarParams.SafetyModel.fisker, param)
    self.safety.init_tests()

  def _gas(self, raw):
    dat = bytearray(16)
    dat[6] = raw
    return self.safety.safety_rx_hook(common.make_msg(0, GAS_MSG, dat=bytes(dat)))

  def _vcu_cruise(self, state):
    dat = bytearray(8)
    dat[4] = state
    self.safety.safety_rx_hook(common.make_msg(0, 0x358, dat=bytes(dat)))

  def test_gas_with_long(self):
    self._init(FiskerSafetyFlags.LONG_CONTROL)
    self._vcu_cruise(3)
    self.assertTrue(self.safety.get_controls_allowed())
    self.assertTrue(self._gas(250))
    self.assertTrue(self.safety.get_gas_pressed_prev())
    self.assertTrue(self.safety.get_controls_allowed())
    self.assertFalse(self.safety.get_longitudinal_allowed())
    self.assertTrue(self._gas(0))
    self.assertTrue(self.safety.get_longitudinal_allowed())

  def test_gas_ignored_without_long(self):
    self._init(0)
    self._vcu_cruise(3)
    self.assertTrue(self.safety.get_controls_allowed())
    self._gas(250)
    self.assertFalse(self.safety.get_gas_pressed_prev())
    self.assertTrue(self.safety.get_longitudinal_allowed())


if __name__ == "__main__":
  unittest.main()
