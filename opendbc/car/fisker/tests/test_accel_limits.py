"""
The jerk (slew) limiter on the 0x121 accel request (apply_fisker_accel_limits), alone and through the
carcontroller: the request goes to the inactive 0.0 at once when not commanding, starts from the
measured acceleration on the first commanding frame (no step at engage or after a gas override), then
moves toward openpilot's accel by at most ACCEL_JERK_UP / ACCEL_JERK_DOWN per 10 ms frame, and is
never outside ACCEL_MIN..ACCEL_MAX. The limits come from the stock ADAS's own requests (see values.py).
"""
import random
import unittest

from opendbc.car import DT_CTRL, structs
from opendbc.car.fisker.carcontroller import apply_fisker_accel_limits
from opendbc.car.fisker.tests.test_long_frames import decode, make_long_car
from opendbc.car.fisker.values import CANBUS, CarControllerParams as P

UP = P.ACCEL_JERK_UP * DT_CTRL      # largest rise per frame
DOWN = P.ACCEL_JERK_DOWN * DT_CTRL  # largest fall per frame
EPS = 1e-9
LSB = 0.0004882                     # 0x121 AccelReq resolution


class TestAccelLimitsValues(unittest.TestCase):
  def test_limits_at_or_below_stock(self):
    # Stock ADAS 0x121 while stock ACC Active, over 100 ms windows: rising p99 1.50, falling p99 2.80 m/s^3.
    assert 0 < P.ACCEL_JERK_UP <= 1.0
    assert 0 < P.ACCEL_JERK_DOWN <= 2.8


class TestApplyAccelLimits(unittest.TestCase):
  def test_inactive_is_zero_at_once(self):
    for last in (-3.5, -1.2, 0.0, 0.7, 1.0):
      for was_active in (False, True):
        assert apply_fisker_accel_limits(-2.0, last, 0.5, False, was_active) == 0.0

  def test_first_active_frame_is_measured(self):
    assert apply_fisker_accel_limits(-2.0, 0.0, 0.3, True, False) == 0.3
    assert apply_fisker_accel_limits(1.0, 0.0, -0.4, True, False) == -0.4
    # clipped to the request range
    assert apply_fisker_accel_limits(0.0, 0.0, 1.6, True, False) == P.ACCEL_MAX
    assert apply_fisker_accel_limits(0.0, 0.0, -5.0, True, False) == P.ACCEL_MIN

  def test_rise_and_fall_are_limited_separately(self):
    assert abs(apply_fisker_accel_limits(1.0, 0.0, 0.0, True, True) - UP) < EPS
    assert abs(apply_fisker_accel_limits(-3.5, 0.0, 0.0, True, True) + DOWN) < EPS
    # releasing the brake is a rise too
    assert abs(apply_fisker_accel_limits(0.0, -2.0, 0.0, True, True) - (-2.0 + UP)) < EPS
    # a small change inside the limits is passed through
    assert apply_fisker_accel_limits(0.004, 0.0, 0.0, True, True) == 0.004

  def test_target_outside_limits_is_clipped(self):
    last = P.ACCEL_MAX - UP / 2
    assert apply_fisker_accel_limits(5.0, last, 0.0, True, True) == P.ACCEL_MAX
    last = P.ACCEL_MIN + DOWN / 2
    assert apply_fisker_accel_limits(-9.0, last, 0.0, True, True) == P.ACCEL_MIN

  def test_reaches_target(self):
    a = 0.0
    for _ in range(int(1.0 / UP) + 2):
      a = apply_fisker_accel_limits(0.8, a, 0.0, True, True)
    assert a == 0.8
    for _ in range(int(3.0 / DOWN) + 2):
      a = apply_fisker_accel_limits(-2.0, a, 0.0, True, True)
    assert a == -2.0

  def test_random_sequences_stay_inside_limits(self):
    rng = random.Random(0)
    last, was_active = 0.0, False
    for _ in range(20000):
      active = rng.random() > 0.05
      target = rng.uniform(-6.0, 3.0)
      out = apply_fisker_accel_limits(target, last, rng.uniform(-6.0, 3.0), active, was_active)
      assert P.ACCEL_MIN <= out <= P.ACCEL_MAX
      if not active:
        assert out == 0.0
      elif was_active:
        assert -DOWN - EPS <= out - last <= UP + EPS
      last, was_active = out, active


def tick(controller, cs, accel, long_active=True, gas=False):
  """One carcontroller tick with openpilot asking for `accel`: the decoded 0x121, 0x117 and 0x118 sent."""
  cs.out.cruiseState.enabled = True
  cs.out.gasPressed = gas
  cc = structs.CarControl(longActive=long_active)
  cc.actuators.accel = accel
  _, sends = controller.update(cc.as_reader(), structs.CarControlSP(), cs, 0)
  frames = {addr: decode(addr, data) for addr, data, bus in sends if addr in (0x117, 0x118, 0x121) and bus == CANBUS.pt}
  return frames[0x121], frames[0x117], frames[0x118]


class TestCarControllerAccel(unittest.TestCase):
  def test_engage_starts_from_measured_then_slews(self):
    controller, cs = make_long_car()
    cs.a_ego_wheel = 0.3
    req, _, _ = tick(controller, cs, -2.0)
    assert abs(req["ADAS_LgtCtrl_AccelReq"] - 0.3) < LSB
    assert req["ADAS_LgtCtrl_AccelVld"] == 1
    prev = 0.3
    for _ in range(20):
      req, _, _ = tick(controller, cs, -2.0)
      a = req["ADAS_LgtCtrl_AccelReq"]
      assert abs((prev - a) - DOWN) < 2 * LSB
      prev = a

  def test_gas_override_drops_to_inactive_at_once_and_restarts_from_measured(self):
    controller, cs = make_long_car()
    cs.a_ego_wheel = 0.0
    for _ in range(50):
      tick(controller, cs, 1.0)
    assert abs(controller.accel_last - 49 * UP) < 1e-6   # first frame measured (0.0), then 49 rises
    req, status, _ = tick(controller, cs, 1.0, gas=True)
    assert abs(req["ADAS_LgtCtrl_AccelReq"]) < LSB and req["ADAS_LgtCtrl_AccelVld"] == 0
    assert status["ADAS_LgtCtrl_DrvrOvrdSts"] == 1
    cs.a_ego_wheel = 0.6
    req, _, _ = tick(controller, cs, -1.0)
    assert abs(req["ADAS_LgtCtrl_AccelReq"] - 0.6) < LSB

  def test_disengage_is_inactive_at_once(self):
    controller, cs = make_long_car()
    for _ in range(10):
      tick(controller, cs, -3.0)
    req, status, _ = tick(controller, cs, -3.0, long_active=False)
    assert abs(req["ADAS_LgtCtrl_AccelReq"]) < LSB and req["ADAS_LgtCtrl_AccelVld"] == 0
    assert status["ADAS_LgtCtrl_Sts"] == 0

  def test_standstill_hold_frames(self):
    # openpilot's LongControl holds CarParams.stopAccel (default -2.0) once stopped. What goes out: a plain
    # ACC decel request, no standstill / hold request bits (panda rejects those).
    controller, cs = make_long_car()
    cs.out.vEgoRaw = 0.0
    cs.out.standstill = True
    for _ in range(200):
      req, status, esp = tick(controller, cs, -2.0)
    assert abs(req["ADAS_LgtCtrl_AccelReq"] + 2.0) < LSB
    assert req["ADAS_LgtCtrl_AccelVld"] == 1
    assert (status["ADAS_LgtCtrl_Sts"], status["ADAS_LgtCtrl_Typ"]) == (3, 1)
    assert status["ADAS_HAP_EmgyStandstillReq"] == 0 and status["ADAS_ParkStandstillReq"] == 0
    assert esp["ADAS_ESP_StandstillReq"] == 0 and esp["ADAS_LgtCtrl_ESP_Sts"] == 0

  def test_new_actuators_report_the_sent_accel(self):
    controller, cs = make_long_car()
    cs.a_ego_wheel = 0.2
    cc = structs.CarControl(longActive=True)
    cc.actuators.accel = -1.0
    cs.out.cruiseState.enabled = True
    new_actuators, _ = controller.update(cc.as_reader(), structs.CarControlSP(), cs, 0)
    assert abs(new_actuators.accel - 0.2) < 1e-6
