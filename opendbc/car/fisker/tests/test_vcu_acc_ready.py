"""
openpilot longitudinal only: when the VCU withdraws VCU_ACCRdy (VCU_0x214) during a cruise
engagement, carstate reports accFaulted and the carcontroller falls back to the idle long heartbeat
(no accel request). Without openpilot longitudinal the signal is ignored.
"""
import unittest

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus, structs
from opendbc.car.fisker.carcontroller import ACCEL_CAN_ID, LAT_CANDIDATE_BUS, LONG_CANDIDATE_ADDR, CarController
from opendbc.car.fisker.carstate import VCU_ACC_RDY_LOST_FRAMES, CarState
from opendbc.car.fisker.interface import CarInterface
from opendbc.car.fisker.values import CANBUS, CAR, DBC

DBC_NAME = DBC[CAR.FISKER_OCEAN][Bus.pt]
ACTIVE, STANDBY = 3, 2


def make_carstate(op_long: bool):
  CP = CarInterface.get_non_essential_params(CAR.FISKER_OCEAN)
  CP.openpilotLongitudinalControl = op_long
  CP.secOcKeyAvailable = True
  CP_SP = CarInterface.get_non_essential_params_sp(CP, CAR.FISKER_OCEAN)
  CS = CarState(CP, CP_SP)
  return CP, CP_SP, CS, CS.get_can_parsers(CP, CP_SP), CANPacker(DBC_NAME)


def feed(CS, parsers, packer, steps):
  """One carState update per (ADAS_Sts_ACC_ICC, VCU_ACCRdy) step, in D. Returns (accFaulted, vcu_acc_rdy_lost) each."""
  out = []
  for acc_state, rdy in steps:
    frames = [
      packer.make_can_msg("ADAS_0x313", CANBUS.cam, {"ADAS_Sts_ACC_ICC": acc_state}),
      packer.make_can_msg("VCU_0x214", CANBUS.pt, {"VCU_GearSig": 4, "VCU_ACCRdy": rdy}),
    ]
    for cp in parsers.values():
      cp.update([(0, frames)])
    ret, _ = CS.update(parsers)
    out.append((ret.accFaulted, CS.vcu_acc_rdy_lost))
  return out


def long_frames(CP, CP_SP, CS, long_active=True, accel=0.5):
  """What the carcontroller commands for this carstate: (AccelVld, AccelReq) of its 0x121 candidates, (0, 0.0) if
  it sends none (then the OEM's own 0x121 reaches the VCU untouched)."""
  controller = CarController(DBC[CAR.FISKER_OCEAN], CP, CP_SP)
  controller.secoc_key_verified = True
  controller.secoc_key = bytes(16)  # synthetic key, no vehicle credentials
  CS.out = structs.CarState(vEgoRaw=10.)
  CS.out.cruiseState.enabled = True
  CS.oem_121_vld = 1   # the OEM ACC is commanding
  cc = structs.CarControl(longActive=long_active)
  cc.actuators.accel = accel
  CS.a_ego_wheel = accel  # a fresh controller's first request is the measured accel (jerk limiter); make it the request
  _, sends = controller.update(cc.as_reader(), structs.CarControlSP(), CS, 0)
  cands = [m for m in sends if m[0] == LONG_CANDIDATE_ADDR and m[2] == LAT_CANDIDATE_BUS]
  if not cands:
    return 0, 0.0
  parser = CANParser(DBC_NAME, [("ADAS_0x121", float('nan'))], CANBUS.pt)
  parser.update([(0, [(ACCEL_CAN_ID, cands[0][1], CANBUS.pt)])])
  cmd = parser.vl["ADAS_0x121"]
  return int(cmd["ADAS_LgtCtrl_AccelVld"]), round(cmd["ADAS_LgtCtrl_AccelReq"], 2)


class TestVcuAccReady(unittest.TestCase):
  def test_drop_while_engaged_faults(self):
    _, _, CS, parsers, packer = make_carstate(op_long=True)
    assert feed(CS, parsers, packer, [(ACTIVE, 1)] * 3) == [(False, False)] * 3
    out = feed(CS, parsers, packer, [(ACTIVE, 0)] * (VCU_ACC_RDY_LOST_FRAMES + 2))
    assert out[:VCU_ACC_RDY_LOST_FRAMES - 1] == [(False, False)] * (VCU_ACC_RDY_LOST_FRAMES - 1)
    assert out[VCU_ACC_RDY_LOST_FRAMES - 1:] == [(True, True)] * 3

  def test_single_bad_frame_ignored(self):
    _, _, CS, parsers, packer = make_carstate(op_long=True)
    steps = [(ACTIVE, 1)] + ([(ACTIVE, 0)] * (VCU_ACC_RDY_LOST_FRAMES - 1) + [(ACTIVE, 1)]) * 3
    assert not any(faulted or lost for faulted, lost in feed(CS, parsers, packer, steps))

  def test_clears_when_ready_again_or_cruise_ends(self):
    for recovery in ((ACTIVE, 1), (STANDBY, 0)):
      _, _, CS, parsers, packer = make_carstate(op_long=True)
      feed(CS, parsers, packer, [(ACTIVE, 1)])
      assert feed(CS, parsers, packer, [(ACTIVE, 0)] * VCU_ACC_RDY_LOST_FRAMES)[-1] == (True, True)
      assert feed(CS, parsers, packer, [recovery])[-1] == (False, False), recovery

  def test_not_ready_without_a_drop_is_not_a_fault(self):
    # Not_ready while cruise is off, or from the first frame of an engagement, is not a drop: the
    # ACC failure states and the gear checks cover those.
    _, _, CS, parsers, packer = make_carstate(op_long=True)
    steps = [(STANDBY, 0)] * 20 + [(ACTIVE, 0)] * 20 + [(STANDBY, 0)] * 5 + [(ACTIVE, 0)] * 20
    assert not any(faulted or lost for faulted, lost in feed(CS, parsers, packer, steps))
    # each engagement needs its own Ready: one from before a cancel doesn't carry over
    steps = [(ACTIVE, 1)] * 3 + [(STANDBY, 1)] * 3 + [(ACTIVE, 0)] * 20
    assert not any(faulted or lost for faulted, lost in feed(CS, parsers, packer, steps))

  def test_no_effect_without_openpilot_longitudinal(self):
    CP, CP_SP, CS, parsers, packer = make_carstate(op_long=False)
    steps = [(ACTIVE, 1)] * 3 + [(ACTIVE, 0)] * 50
    assert not any(faulted or lost for faulted, lost in feed(CS, parsers, packer, steps))
    # and nothing longitudinal is sent at all
    controller = CarController(DBC[CAR.FISKER_OCEAN], CP, CP_SP)
    controller.secoc_key_verified = True
    controller.secoc_key = bytes(16)
    CS.out = structs.CarState(vEgoRaw=10.)
    CS.out.cruiseState.enabled = True
    _, sends = controller.update(structs.CarControl(longActive=True).as_reader(), structs.CarControlSP(), CS, 0)
    assert not [msg for msg in sends if msg[0] in (0x117, 0x118, ACCEL_CAN_ID)]

  def test_carcontroller_stops_commanding_accel(self):
    CP, CP_SP, CS, parsers, packer = make_carstate(op_long=True)
    feed(CS, parsers, packer, [(ACTIVE, 1)] * 3)
    assert long_frames(CP, CP_SP, CS) == (1, 0.5)   # Valid, the request
    feed(CS, parsers, packer, [(ACTIVE, 0)] * VCU_ACC_RDY_LOST_FRAMES)
    assert CS.vcu_acc_rdy_lost
    assert long_frames(CP, CP_SP, CS) == (0, 0.0)   # no candidates: the OEM's own 0x121
    assert long_frames(CP, CP_SP, CS, long_active=False) == (0, 0.0)
    feed(CS, parsers, packer, [(ACTIVE, 1)])
    assert long_frames(CP, CP_SP, CS) == (1, 0.5)


if __name__ == "__main__":
  unittest.main()
