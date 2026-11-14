"""
The 0x121 accel frame openpilot injects (create_accel_command) must encode accel on the stock
ADAS module's scale, carry a valid plain-E2E checksum, and signal AccelVld only while openpilot is
actually commanding longitudinal — an idle heartbeat (accel_active=False) sends AccelVld=Init so the
VCU ignores the request. Scale/offset are the DBC's for ADAS_LgtCtrl_AccelReq; the golden raw values
come from stock frames in drives 000000af--28a9f95752--1 and 000000b4--d0f733ebb2--5. The 0x117/0x118
status values per state match what the stock ADAS sends, and the long counters (AliveCounters and
0x121's SecOC counter) continue the OEM ADAS's own bus-2 streams at the first long frame, then free-run.
"""
import unittest

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus, structs
from opendbc.car.fisker.carcontroller import CarController
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
  return controller, cs


def long_counters(controller, cs, cruise=True, long_active=True):
  """One carcontroller tick: {addr: AliveCounter} of the long frames sent, plus 'secoc' = 0x121's wire counter."""
  cs.out.cruiseState.enabled = cruise
  _, sends = controller.update(structs.CarControl(longActive=long_active).as_reader(), structs.CarControlSP(), cs, 0)
  out = {addr: data[1] & 0xF for addr, data, bus in sends if addr in (0x121, 0x117, 0x118) and bus == CANBUS.pt}
  for addr, data, _bus in sends:
    if addr == 0x121:
      out['secoc'] = (data[4] >> 2) & 0x3F
  return out


def sent_0x121_freshness(controller, cs):
  """One long-active tick: (reset, secoc counter) of the 0x121 sent. The wire byte carries the reset flag and
  the low 6 bits of the counter; the MAC is checked against the full values, so both must be right."""
  cs.out.cruiseState.enabled = True
  _, sends = controller.update(structs.CarControl(longActive=True).as_reader(), structs.CarControlSP(), cs, 0)
  data = next(d for addr, d, bus in sends if addr == 0x121 and bus == CANBUS.pt)
  reset, ctr = controller.secoc_reset_prev, controller.secoc_window_ctr
  assert data[4] == wire_freshness_byte(ctr, reset)
  assert data[5:8] == secoc_mac(controller.secoc_key, 0x121, data[:4], cs.secoc_trip, reset, ctr)
  return reset, ctr


class TestLongCounters(unittest.TestCase):
  def test_continue_oem_at_first_frame_then_free_run_skipping_15(self):
    controller, cs = make_long_car()
    cs.oem_121_alive, cs.oem_117_alive, cs.oem_118_alive = 13, 14, 5
    cs.oem_121_secoc_full_ctr = 40
    assert long_counters(controller, cs, cruise=False, long_active=False) == {}   # not engaged yet: nothing sent

    # First long frame: OEM_last + 1 for every counter (14 + 1 wraps to 0: 15 is never sent).
    assert long_counters(controller, cs) == {0x121: 14, 0x117: 0, 0x118: 6, 'secoc': 41}
    # OEM's counters keep moving on bus 2 but we free-run from our own, +1 per frame.
    cs.oem_121_alive, cs.oem_117_alive, cs.oem_118_alive = 2, 2, 2
    cs.oem_121_secoc_full_ctr = 7
    expected = [(0, 1, 7, 42), (1, 2, 8, 43), (2, 3, 9, 44)]
    for a121, a117, a118, secoc in expected:
      assert long_counters(controller, cs) == {0x121: a121, 0x117: a117, 0x118: a118, 'secoc': secoc}

  def test_counters_never_reach_15(self):
    controller, cs = make_long_car()
    seen = set()
    for _ in range(40):
      c = long_counters(controller, cs)
      seen |= {c[0x121], c[0x117], c[0x118]}
    assert seen == set(range(15))

  def test_reset_restarts_0x121_secoc_counter_when_oem_does(self):
    # Frame by frame around a GW reset step. GW_Syn_All steps first; OEM's 0x121 restarts its SecOC
    # counter 0-10 ms later (one frame here), which carstate sees as 0x121's reset flag flipping. Our
    # 0x121 stays in the old window until then and restarts at 1 on the same frame as OEM, the same
    # rule as the 0x1D0 candidates (oem_reset_value).
    controller, cs = make_long_car()
    cs.oem_121_secoc_full_ctr = 98
    assert sent_0x121_freshness(controller, cs) == (5, 99)   # seed: OEM + 1, in OEM's window
    assert sent_0x121_freshness(controller, cs) == (5, 100)
    cs.secoc_reset = 6                                          # GW steps; OEM still in window 5
    assert sent_0x121_freshness(controller, cs) == (5, 101)
    cs.oem_121_reset_flag = 6 & 0x3                             # OEM restarts one frame later
    assert sent_0x121_freshness(controller, cs) == (6, 1)
    assert sent_0x121_freshness(controller, cs) == (6, 2)

  def test_restart_on_the_gw_frame_when_oem_has_no_lag(self):
    # A 0.0 ms lag (also seen on the car): OEM's flag flips on the same frame as GW_Syn_All.
    controller, cs = make_long_car()
    cs.oem_121_secoc_full_ctr = 60
    assert sent_0x121_freshness(controller, cs) == (5, 61)
    cs.secoc_reset, cs.oem_121_reset_flag = 6, 6 & 0x3
    assert sent_0x121_freshness(controller, cs) == (6, 1)

  def test_seed_right_after_gw_step_uses_oem_window(self):
    # First long frame between the GW step and OEM's restart: the seed counter is OEM's (old window),
    # so the frame is stamped with the old reset, not the GW's new one.
    controller, cs = make_long_car()
    cs.secoc_reset, cs.oem_121_reset_flag = 6, 5 & 0x3
    cs.oem_121_secoc_full_ctr = 100
    assert sent_0x121_freshness(controller, cs) == (5, 101)
    cs.oem_121_reset_flag = 6 & 0x3
    assert sent_0x121_freshness(controller, cs) == (6, 1)

  def test_window_never_goes_back(self):
    # A stale or glitched OEM flag must not make us stamp an older reset window again (a replay).
    controller, cs = make_long_car()
    cs.oem_121_secoc_full_ctr = 10
    assert sent_0x121_freshness(controller, cs) == (5, 11)
    cs.secoc_reset, cs.oem_121_reset_flag = 6, 6 & 0x3
    assert sent_0x121_freshness(controller, cs) == (6, 1)
    cs.oem_121_reset_flag = 5 & 0x3
    assert sent_0x121_freshness(controller, cs) == (6, 2)

  def test_resnap_after_tx_stops(self):
    # Long TX stops only while it was never long-active (the heartbeat stays on after that); the
    # next start continues OEM again.
    controller, cs = make_long_car()
    cs.oem_121_alive, cs.oem_117_alive, cs.oem_118_alive = 3, 4, 5
    assert long_counters(controller, cs, long_active=False) == {0x121: 4, 0x117: 5, 0x118: 6, 'secoc': 1}
    assert long_counters(controller, cs, cruise=False, long_active=False) == {}
    cs.oem_121_alive, cs.oem_117_alive, cs.oem_118_alive = 9, 10, 11
    cs.oem_121_secoc_full_ctr = 20
    assert long_counters(controller, cs) == {0x121: 10, 0x117: 11, 0x118: 12, 'secoc': 21}
