#!/usr/bin/env python3
import unittest

from opendbc.car.fisker.values import FiskerSafetyFlags
from opendbc.car.structs import CarParams
import opendbc.safety.tests.common as common
from opendbc.safety.tests.libsafety import libsafety_py

GAS_MSG = 0x214      # VCU_0x214, 16 bytes
ACCEL_MSG = 0x121
LONG_MSGS = (0x121, 0x117, 0x118)
# the messages every Fisker configuration needs to see: (addr, bus)
COMMON_RX_MSGS = ((0x115, 0), (0x318, 0), (0x1C2, 0), (0x1C4, 0), (0x313, 2), (0x514, 0))


class TestFiskerLongFlagReleaseBuild(unittest.TestCase):
  """Release firmware (no ALLOW_DEBUG) ignores the longitudinal flag."""

  @classmethod
  def setUpClass(cls):
    cls.release = libsafety_py.ffi.dlopen(libsafety_py._build_libsafety(release=True))

  def setUp(self):
    self.release.set_timer(0)
    self.release.set_current_safety_param_sp(0)
    self.release.set_safety_hooks(CarParams.SafetyModel.fisker, FiskerSafetyFlags.LONG_CONTROL)
    self.release.init_tests()

  def _accel_dat(self, raw):
    dat = bytearray(8)
    dat[2], dat[3] = raw >> 8, raw & 0xFF
    return bytes(dat)

  def test_long_frames_not_sent(self):
    for controls_allowed in (False, True):
      self.release.set_controls_allowed(controls_allowed)
      for raw in (32773, 34822, 25604):   # inactive, +1.0, -3.5 m/s^2
        self.assertFalse(self.release.safety_tx_hook(common.make_msg(0, ACCEL_MSG, dat=self._accel_dat(raw))))
      for addr in (0x117, 0x118):
        self.assertFalse(self.release.safety_tx_hook(common.make_msg(0, addr, 8)))

  def test_oem_long_frames_always_forwarded(self):
    self.release.set_controls_allowed(True)
    for addr in LONG_MSGS:
      self.release.safety_tx_hook(common.make_msg(0, addr, dat=self._accel_dat(32773)))
      self.assertEqual(self.release.safety_fwd_hook(2, addr), 0)
      self.assertEqual(self.release.safety_fwd_hook(0, addr), 2)

  def test_gas_message_not_an_rx_check(self):
    dat = bytearray(16)
    dat[6] = 250
    self.release.safety_rx_hook(common.make_msg(0, GAS_MSG, dat=bytes(dat)))
    self.assertFalse(self.release.get_gas_pressed_prev())
    for addr, bus in COMMON_RX_MSGS:
      self.assertTrue(self.release.safety_rx_hook(common.make_msg(bus, addr, 8)))
    self.release.safety_tick_current_safety_config()
    self.assertTrue(self.release.safety_config_valid())

  def test_steering_still_sent(self):
    self.release.set_controls_allowed(True)
    self.release.set_desired_angle_last(0)
    self.assertTrue(self.release.safety_tx_hook(common.make_msg(0, 0x1D0, dat=self._accel_dat(12480))))
    self.assertTrue(self.release.safety_tx_hook(common.make_msg(0, 0x1C0, 8)))


if __name__ == "__main__":
  unittest.main()
