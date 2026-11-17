"""
The 0x121 accel frame (create_accel_command) must encode accel on the stock ADAS module's scale and carry a valid
plain-E2E checksum; the golden raw values come from stock frames in drives 000000af--28a9f95752--1 and
000000b4--d0f733ebb2--5. openpilot longitudinal is modify-and-forward: it never sends 0x121 / 0x117 / 0x118 of its
own, only 0x121 candidates (0x521) carrying OEM's own alive / SecOC counters and reserved bits, for the panda to
swap into the OEM's frame. On the car (verify_121, 2026-11-14) the candidates rebuilt the OEM's own 0x121 byte for
byte, MAC included, for 11509 of 11537 frames; the rest fall at SecOC window restarts, where the OEM frame passes.
"""
import unittest

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus, structs
from opendbc.car.fisker.carcontroller import LAT_CANDIDATE_BUS, LONG_CANDIDATE_ADDR, CarController, lat_candidates
from opendbc.car.fisker.carstate import CarState
from opendbc.car.fisker.fiskercan import FiskerCAN, fisker_plain_checksum
from opendbc.car.fisker.interface import CarInterface
from opendbc.car.fisker.secoc import secoc_mac, wire_freshness_byte
from opendbc.car.fisker.values import CANBUS, CAR, DBC
from opendbc.testing import parameterized


def make():
  return FiskerCAN(None, CANPacker(DBC[CAR.FISKER_OCEAN][Bus.pt]))


class TestLongFrames(unittest.TestCase):
  def test_0x121_zero_accel_matches_stock_raw(self):
    # accel 0.0 encodes to the stock idle raw value 0x8005 in bytes 2-3.
    _, data, _ = make().create_accel_command(0.0, True, 7)
    assert ((data[2] << 8) | data[3]) == 0x8005
    assert data[1] & 0xF == 7                      # AliveCounter is byte1 low nibble

  def test_0x121_accel_scales_on_the_dbc_factor(self):
    for accel in (-3.5, -0.85, 0.5, 2.0):
      d = make().create_accel_command(accel, True, 0)[1]
      raw = (d[2] << 8) | d[3]
      assert abs(raw * 0.0004882 - 16 - accel) < 0.001

  def test_0x121_accel_vld_follows_active(self):
    # AccelVld (byte1 bits 4-5): 1(Valid) while commanding, 0(Init) on the idle heartbeat.
    assert (make().create_accel_command(0.0, True, 0)[1][1] >> 4) & 0x3 == 1
    assert (make().create_accel_command(0.0, False, 0)[1][1] >> 4) & 0x3 == 0

  def test_checksum_valid(self):
    addr, data, _ = make().create_accel_command(1.0, True, 3)
    assert fisker_plain_checksum(addr, bytes([0]) + data[1:]) == data[0]

  def test_accel_cap_matches_panda(self):
    from opendbc.car.fisker.interface import CarInterface
    from opendbc.car.fisker.values import CarControllerParams
    assert CarControllerParams.ACCEL_MAX == 1.0
    assert CarInterface.get_pid_accel_limits(None, None, 10.0, 20.0) == (CarControllerParams.ACCEL_MIN, 1.0)
    d = make().create_accel_command(CarControllerParams.ACCEL_MAX, True, 0)[1]
    assert (d[2] << 8) | d[3] == 34822   # FISKER_ACCEL_MAX in safety/modes/fisker.h

  def test_panda_accel_limits_match_python(self):
    # Read the panda's own #defines, so its cap can't drift from ours again: it was left at +2.0 m/s^2 after the
    # Python cap dropped to 1.0, and nothing failed (the safety test asserted the C value, this one the Python).
    import os
    import re
    from opendbc.car.fisker.values import CarControllerParams
    header = os.path.join(os.path.dirname(__file__), "..", "..", "..", "safety", "modes", "fisker.h")
    with open(header) as f:
      defines = dict(re.findall(r"#define (FISKER_ACCEL_\w+)\s+(\d+)", f.read()))
    fc = make()
    for name, accel, active in (("FISKER_ACCEL_MAX", CarControllerParams.ACCEL_MAX, True),
                                ("FISKER_ACCEL_MIN", CarControllerParams.ACCEL_MIN, True),
                                ("FISKER_ACCEL_INACTIVE", 0.0, False)):
      d = fc.create_accel_command(accel, active, 0)[1]
      assert int(defines[name]) == (d[2] << 8) | d[3], name


def decode(addr, data):
  name = {0x117: "ADAS_0x117", 0x118: "ADAS_0x118", 0x121: "ADAS_0x121"}[addr]
  parser = CANParser(DBC[CAR.FISKER_OCEAN][Bus.pt], [(name, float('nan'))], CANBUS.pt)
  parser.update([(0, [(addr, data, CANBUS.pt)])])
  return parser.vl[name]


class TestLongFramesMatchStock(unittest.TestCase):
  # What the stock ADAS sends (OEM bus-2 frames in 5 recorded drives). 0x117 (Sts, Typ):
  # ACC Active, braking included, and ACC Override both carry (3, 1); Typ=2 never appears.
  # 0x118 keeps ESP_Sts=0, StandstillReq=0, StandstillVld=1, JerkReq=0 in every state.
  @parameterized("long_active, gas, sts, typ", [
    (True, False, 3, 1),    # openpilot commanding = stock ACC Active
    (True, True, 3, 1),     # driver on the gas = stock ACC Override
    (False, False, 0, 0),   # idle heartbeat = stock ACC off
  ])
  def test_0x117_and_0x118_match_stock(self, long_active, gas, sts, typ):
    fc = make()
    status = decode(*fc.create_long_status(long_active, gas, gear_req=0, counter=4)[:2])
    assert (status["ADAS_LgtCtrl_Sts"], status["ADAS_LgtCtrl_Typ"]) == (sts, typ)
    assert status["ADAS_LgtCtrl_DrvrOvrdSts"] == int(gas)
    esp = decode(*fc.create_long_esp_handshake(long_active, gas, counter=4)[:2])
    assert esp["ADAS_LgtCtrl_ESP_Sts"] == 0
    assert esp["ADAS_ESP_StandstillReq"] == 0
    assert esp["ADAS_ESP_StandstillVld"] == 1
    assert esp["ADAS_JerkReq"] == 0

  def test_acc_stop_and_go_never_sent(self):
    fc = make()
    for long_active in (False, True):
      for gas in (False, True):
        assert decode(*fc.create_long_status(long_active, gas, 0, 0)[:2])["ADAS_LgtCtrl_Typ"] != 2


def make_long_car():
  cp = CarInterface.get_non_essential_params(CAR.FISKER_OCEAN)
  cp.openpilotLongitudinalControl = True
  cp.secOcKeyAvailable = True
  sp = CarInterface.get_non_essential_params_sp(cp, CAR.FISKER_OCEAN)
  cs = CarState(cp, sp)
  controller = CarController(DBC[CAR.FISKER_OCEAN], cp, sp)
  controller.secoc_key_verified = True
  controller.secoc_key = bytes(16)  # synthetic key, no vehicle credentials
  cs.out = structs.CarState(vEgoRaw=10.)
  cs.secoc_trip, cs.secoc_reset = 1, 5
  cs.oem_121_reset_flag = 5 & 0x3   # OEM's 0x121 is in the GW's window
  cs.oem_121_secoc_full_ctr, cs.oem_121_alive = 40, 7
  cs.oem_121_vld, cs.oem_121_rsv = 1, 0   # the OEM ACC is engaged and commanding
  cs.adas_acc_state = 3                   # ADAS_Sts_ACC_ICC Active
  return controller, cs


def long_sends(controller, cs, accel=0.5, cruise=True, long_active=True, gas=False):
  """One carcontroller tick: (0x121 candidates as (addr, data, bus), every frame sent)."""
  cs.out.cruiseState.enabled = cruise
  cs.out.gasPressed = gas
  cc = structs.CarControl(longActive=long_active)
  cc.actuators.accel = accel
  _, sends = controller.update(cc.as_reader(), structs.CarControlSP(), cs, 0)
  return [m for m in sends if m[0] == LONG_CANDIDATE_ADDR and m[2] == LAT_CANDIDATE_BUS], sends


class TestLongCandidates(unittest.TestCase):
  """openpilot longitudinal is modify-and-forward: never a frame of its own on 0x121 / 0x117 / 0x118, only complete
  0x121 frames (the candidates) for the panda to swap into the OEM's own 0x121 (safety fisker_fwd_modify)."""

  def test_never_sends_the_oem_long_messages(self):
    controller, cs = make_long_car()
    for kw in ({}, {"gas": True}, {"long_active": False}, {"cruise": False}):
      _, sends = long_sends(controller, cs, **kw)
      assert not [m for m in sends if m[0] in (0x121, 0x117, 0x118)], kw

  def test_candidates_continue_oem_counters_and_verify(self):
    controller, cs = make_long_car()
    cands, _ = long_sends(controller, cs)
    assert cands
    slots = []
    for _addr, data, _bus in cands:
      ctr = next(k for k, a, r in lat_candidates(40, 7, 5) if wire_freshness_byte(k, r) == data[4] and a == data[1] & 0xF)
      assert data[5:8] == secoc_mac(controller.secoc_key, 0x121, data[:4], cs.secoc_trip, 5, ctr)   # a valid MAC
      assert fisker_plain_checksum(0x121, data) == data[0]
      slots.append(ctr)
    assert 40 in slots and 41 in slots     # OEM's current frame and the next one are covered
    for ctr, (_addr, data, _bus) in zip(slots, cands, strict=True):
      assert data[1] & 0xF == (7 + ctr - 40) % 15   # alive steps with the counter, OEM's numbering

  def test_candidate_carries_our_accel_valid_and_oem_reserved_bits(self):
    controller, cs = make_long_car()
    cs.oem_121_rsv = 2
    cs.a_ego_wheel = 0.5
    cands, _ = long_sends(controller, cs, accel=0.5)
    for _addr, data, _bus in cands:
      d = decode(0x121, data)
      assert abs(d["ADAS_LgtCtrl_AccelReq"] - 0.5) < 0.001 and d["ADAS_LgtCtrl_AccelVld"] == 1
      assert d["ADAS_121_Rsv_B1"] == 2

  def test_no_candidates_unless_oem_acc_commanding_and_driver_off_gas(self):
    for setup, kw in ((lambda cs: setattr(cs, "oem_121_vld", 0), {}),   # OEM ACC not commanding
                      (lambda cs: None, {"gas": True}),                    # driver on the gas
                      (lambda cs: None, {"long_active": False})):          # openpilot not commanding
      controller, cs = make_long_car()
      setup(cs)
      cands, _ = long_sends(controller, cs, **kw)
      assert not cands, kw

  def test_only_while_the_oem_acc_is_active_or_holding_at_standstill(self):
    # Standby (2) still carries AccelVld=1 in the OEM frame; Override (4), Deactivation (7, 8) and Failure (9, 10) are
    # the OEM's too. Active (3) and the standstill states (5, 6, 11) are ours: a stop openpilot made stays held.
    for state in range(12):
      controller, cs = make_long_car()
      cs.adas_acc_state = state
      cands, _ = long_sends(controller, cs)
      assert bool(cands) == (state in (3, 5, 6, 11)), state

  def test_handover_logged_once_per_change(self):
    import logging
    from opendbc.car.carlog import carlog
    controller, cs = make_long_car()
    lines = []

    class Grab(logging.Handler):
      def emit(self, record):
        if isinstance(record.msg, dict) and record.msg.get("event") == "fisker_long_handover":
          lines.append(record.msg)
    h = Grab()
    carlog.addHandler(h)
    try:
      for _ in range(5):
        long_sends(controller, cs)
      long_sends(controller, cs, gas=True)
      long_sends(controller, cs, gas=True)
      cs.adas_acc_state = 2
      long_sends(controller, cs)
    finally:
      carlog.removeHandler(h)
    assert [(x["commanding"], x["gas"], x["acc_state"]) for x in lines] == [(True, False, 3), (False, True, 3), (False, False, 2)]
