#!/usr/bin/env python3
"""
Panda-side proof for Fisker openpilot longitudinal as modify-and-forward of the OEM ADAS's own 0x121.

openpilot never sends 0x121 / 0x117 / 0x118 of its own (building them from scratch made the VCU withdraw
VCU_ACCRdy ~50 ms into every engage, route 000000ea--710cc8b5b6). It ships fully-stamped 0x121 payloads as 0x521
on bus 1; panda buffers them (never transmits them) keyed by byte 4 (SecOC wire counter + reset flag) and the E2E
alive counter, and when the OEM's 0x121 is forwarded bus 2 -> bus 0 swaps in the matching candidate, only while:
the OEM frame carries a valid request (AccelVld = 1: its ACC is commanding), the reserved bits match, the candidate's
accel passed the limits, and longitudinal is allowed (controls allowed, gas not pressed). In every other case the
OEM's frame goes through unchanged: never dropped, so the worst case is stock ACC.
"""
import unittest

from opendbc.car.fisker.values import FiskerSafetyFlags
from opendbc.car.structs import CarParams
from opendbc.safety.tests.libsafety import libsafety_py

LONG = FiskerSafetyFlags.LONG_CONTROL
FRESH_US = 60_000
ACCEL_LSB, ACCEL_OFFSET = 0.0004882, -16.0


def accel_raw(accel: float) -> int:
  return round((accel - ACCEL_OFFSET) / ACCEL_LSB)


def _frame(ctr, accel, alive, rflag, rsv, vld, tag):
  """An 0x121-shaped payload: byte 1 = Rsv(7..6) | AccelVld(5..4) | alive(3..0); bytes 2-3 AccelReq; byte 4 =
  (wire counter << 2) | reset flag; bytes 5-7 stand in for the MAC, with a tag telling whose frame it is."""
  raw = accel_raw(accel)
  return bytes([0x5A, ((rsv & 3) << 6) | ((vld & 3) << 4) | (alive & 0xF), (raw >> 8) & 0xFF, raw & 0xFF,
                (((ctr & 0x3F) << 2) | (rflag & 3)) & 0xFF, tag, tag, tag])


def candidate(ctr, accel=0.5, alive=None, rflag=1, rsv=0, vld=1):
  return _frame(ctr, accel, ctr % 15 if alive is None else alive, rflag, rsv, vld, 0xC0)


def send_candidate(safety, dat) -> bool:
  return safety.safety_tx_hook(libsafety_py.make_CANPacket(0x521, 1, bytes(dat)))


def oem_frame(safety, ctr, accel=-0.3, alive=None, rflag=1, rsv=0, vld=1):
  """Forward the OEM's 0x121 for counter `ctr` through fisker_fwd_modify: (emit, the packet as forwarded)."""
  pkt = libsafety_py.make_CANPacket(0x121, 2, _frame(ctr, accel, ctr % 15 if alive is None else alive, rflag, rsv, vld, 0x0E))
  return safety.safety_fwd_modify_hook(pkt), pkt


class TestFiskerLongCandidate(unittest.TestCase):
  def setUp(self):
    self.safety = libsafety_py.libsafety
    self._init(LONG)

  def _init(self, param):
    self.safety.set_timer(0)
    self.safety.set_safety_hooks(CarParams.SafetyModel.fisker, param)
    self.safety.init_tests()
    self.safety.set_controls_allowed(True)
    self.safety.set_timer(1_000_000)

  def _forwarded(self, ctr, **oem):
    emit, pkt = oem_frame(self.safety, ctr, **oem)
    self.assertTrue(emit, "the OEM's 0x121 must never be dropped")
    return bytes(pkt.data[i] for i in range(8))

  def test_candidates_never_transmitted(self):
    for ctr in range(10, 15):
      self.assertFalse(send_candidate(self.safety, candidate(ctr)), "buffered, never put on bus 1 (the radar link)")

  def test_matching_candidate_replaces_oem_frame(self):
    for ctr in range(10, 15):
      send_candidate(self.safety, candidate(ctr, accel=0.5))
    self.assertEqual(self._forwarded(12), candidate(12, accel=0.5))

  def test_no_candidate_forwards_oem_unchanged(self):
    for ctr in range(10, 15):
      send_candidate(self.safety, candidate(ctr))
    self.assertEqual(self._forwarded(20), _frame(20, -0.3, 20 % 15, 1, 0, 1, 0x0E))   # not buffered: OEM's own
    self.assertEqual(self._forwarded(12, rflag=2)[7], 0x0E)                             # another reset window
    self.assertEqual(self._forwarded(12, alive=3)[7], 0x0E)                             # another alive counter

  def test_oem_acc_not_commanding_is_never_modified(self):
    send_candidate(self.safety, candidate(12))
    for vld in (0, 2, 3):
      self.assertEqual(self._forwarded(12, vld=vld)[7], 0x0E, vld)
    self.assertEqual(self._forwarded(12)[7], 0xC0)

  def test_reserved_bits_must_match(self):
    send_candidate(self.safety, candidate(12, rsv=0))
    self.assertEqual(self._forwarded(12, rsv=2)[7], 0x0E)
    send_candidate(self.safety, candidate(12, rsv=2))
    self.assertEqual(self._forwarded(12, rsv=2)[7], 0xC0)

  def test_candidate_without_valid_request_unusable(self):
    send_candidate(self.safety, candidate(12, vld=0))
    self.assertEqual(self._forwarded(12)[7], 0x0E)

  def test_accel_outside_limits_unusable(self):
    send_candidate(self.safety, candidate(12, accel=3.0))     # above ACCEL_MAX
    self.assertEqual(self._forwarded(12)[7], 0x0E)
    send_candidate(self.safety, candidate(13, accel=-6.0))    # below ACCEL_MIN
    self.assertEqual(self._forwarded(13)[7], 0x0E)

  def test_controls_checked_at_capture_and_at_forward(self):
    self.safety.set_controls_allowed(False)
    send_candidate(self.safety, candidate(12))
    self.safety.set_controls_allowed(True)
    self.assertEqual(self._forwarded(12)[7], 0x0E)            # captured while not allowed: unusable
    send_candidate(self.safety, candidate(13))
    self.safety.set_controls_allowed(False)
    self.assertEqual(self._forwarded(13)[7], 0x0E)            # allowed at capture, not any more: OEM's

  def test_stale_candidate_unusable(self):
    send_candidate(self.safety, candidate(12))
    self.safety.set_timer(1_000_000 + FRESH_US + 1)
    self.assertEqual(self._forwarded(12)[7], 0x0E)

  def test_without_long_flag_nothing_changes(self):
    self._init(0)
    self.assertFalse(send_candidate(self.safety, candidate(12)))
    self.assertEqual(self._forwarded(12)[7], 0x0E)

  def test_long_candidates_do_not_arm_steering_substitution(self):
    # the 0x1D0 pool is separate: with only 0x121 candidates arriving, the OEM's 0x1D0 is forwarded as is
    send_candidate(self.safety, candidate(12))
    pkt = libsafety_py.make_CANPacket(0x1D0, 2, bytes([0, 0, 0x30, 0xC0, 12 << 2, 0xAA, 0, 0]))
    self.assertTrue(self.safety.safety_fwd_modify_hook(pkt))
    self.assertEqual(pkt.data[5], 0xAA)


if __name__ == "__main__":
  unittest.main()
