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
from opendbc.car.fisker.carcontroller import apply_fisker_accel_limits, smooth_accel_step
from opendbc.car.fisker.tests.test_long_frames import decode, long_sends, make_long_car
from opendbc.car.fisker.values import CarControllerParams as P

UP = P.ACCEL_JERK_UP * DT_CTRL      # largest rise per frame
DOWN = P.ACCEL_JERK_DOWN * DT_CTRL  # largest fall per frame
EPS = 1e-9
TOL = 1e-6                          # landing on a moving (filtered) target: < 0.01 m/s^4
LSB = 0.0004882                     # 0x121 AccelReq resolution
SNAP = P.ACCEL_SNAP * DT_CTRL * DT_CTRL   # largest change of the per-frame step, comfort
SNAP_URGENT = P.ACCEL_SNAP_URGENT * DT_CTRL * DT_CTRL
SMOOTH = P.ACCEL_SMOOTH_JERK * DT_CTRL   # largest comfort change per frame


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


def glide(target, a=0.0, step=0.0, n=600, targets=None):
  """Run smooth_accel_step for n frames; the request and per-frame step at each frame."""
  out, filt = [], a
  for i in range(n):
    a, step, filt = smooth_accel_step(targets[i] if targets else target, a, step, filt)
    out.append((a, step))
  return out


class TestAnalogAccel(unittest.TestCase):
  """Driver, 2026-11-17: "avoid quantum jumps, make it feel like analog signal", brake application and release
  continuous. The request eases in and out along an S-curve on top of the jerk limits."""

  def _check_smooth(self, out, step0=0.0, snap=SNAP):
    prev = step0
    for _, step in out:
      assert -max(DOWN, SMOOTH) - EPS <= step <= SMOOTH + EPS
      assert abs(step - prev) <= snap + TOL, (step, prev)
      prev = step

  def test_braking_builds_in_and_settles(self):
    out = glide(-1.5)
    self._check_smooth(out)
    assert abs(out[-1][0] + 1.5) < 1e-9
    assert min(a for a, _ in out) >= -1.5 - 1e-9       # never overshoots into harder braking
    assert abs(out[0][1]) <= SNAP + EPS                 # the first frame barely moves: no corner at the start

  def test_brake_release_fades_out(self):
    out = glide(0.0, a=-2.0)
    self._check_smooth(out)
    assert abs(out[-1][0]) < 1e-9
    assert max(a for a, _ in out) <= 1e-9              # no surge past the target
    # the last 0.2 s before arriving the release is slowing down (the step shrinks), not cut off
    k = next(i for i, (a, _) in enumerate(out) if abs(a) < 1e-9)
    assert out[k - 20][1] > out[k - 2][1] >= 0.0

  def test_drive_builds_in_and_settles(self):
    out = glide(0.8, a=-0.5)
    self._check_smooth(out)
    assert abs(out[-1][0] - 0.8) < 1e-9
    assert max(a for a, _ in out) <= 0.8 + 1e-9

  def test_planner_jitter_is_smoothed(self):
    # openpilot's accel jumping +-0.15 every 50 ms around -1.0
    targets = [-1.0 + (0.15 if (i // 5) % 2 else -0.15) for i in range(600)]
    out = glide(None, a=-1.0, targets=targets)
    tail = [a for a, _ in out[300:]]
    assert max(tail) - min(tail) < 0.03            # the jerk limiter alone: 0.05
    assert abs(sum(tail) / len(tail) + 1.0) < 0.02  # centred on the mean, no ratchet to one side

  def test_urgent_braking_not_delayed(self):
    # a lead braking hard: from cruising to -3.5 asked at once. The jerk limit alone reaches -3.0 in 1.2 s.
    out = glide(-3.5)
    self._check_smooth(out, snap=SNAP_URGENT)
    t = next(i for i, (a, _) in enumerate(out) if a <= -3.0) * DT_CTRL
    assert t <= 1.2 + 0.35, t
    # comfort braking uses the gentle snap
    out = glide(-0.8)
    assert all(abs(s1 - s0) <= SNAP + EPS for (_, s0), (_, s1) in zip(out, out[1:], strict=False))

  def test_old_ramp_had_corners(self):
    # contrast: the jerk limiter alone starts at full rate on the first frame and stops at once at the target
    a = apply_fisker_accel_limits(-1.0, 0.0, 0.0, True, True)
    assert abs(a + DOWN) < EPS
    assert abs(glide(-1.0, n=1)[0][1]) < DOWN / 10

  def test_random_targets_stay_smooth_and_inside_limits(self):
    rng = random.Random(1)
    a, step, filt, target = 0.0, 0.0, 0.0, 0.0
    for i in range(20000):
      if i % 50 == 0:
        target = rng.uniform(-4.0, 1.5)
      a2, step2, filt = smooth_accel_step(target, a, step, filt)
      assert P.ACCEL_MIN - EPS <= a2 <= P.ACCEL_MAX + EPS
      assert -DOWN - EPS <= step2 <= UP + EPS
      if P.ACCEL_MIN + EPS < a2 < P.ACCEL_MAX - EPS:   # not stopped by the request range
        assert abs(step2 - step) <= 1.15 * SNAP_URGENT, (i, target, a, step, a2, step2)   # landing: <= 1.15x
      a, step = a2, step2


def tick(controller, cs, accel, long_active=True, gas=False):
  """One carcontroller tick with openpilot asking for `accel`: the decoded 0x121 candidate (all carry the same
  request), or None when none was sent."""
  cands, _ = long_sends(controller, cs, accel=accel, long_active=long_active, gas=gas)
  return decode(0x121, cands[0][1]) if cands else None


class TestCarControllerAccel(unittest.TestCase):
  def test_engage_starts_from_measured_then_slews(self):
    controller, cs = make_long_car()
    cs.a_ego_wheel = 0.3
    req = tick(controller, cs, -2.0)
    assert abs(req["ADAS_LgtCtrl_AccelReq"] - 0.3) < LSB
    assert req["ADAS_LgtCtrl_AccelVld"] == 1
    # then eases in: each frame's fall grows by at most the comfort snap, never past the jerk limit
    prev, step_prev = 0.3, 0.0
    for _ in range(100):
      req = tick(controller, cs, -2.0)
      a = req["ADAS_LgtCtrl_AccelReq"]
      step = prev - a
      assert -2 * LSB <= step <= DOWN + 2 * LSB
      assert abs(step - step_prev) <= SNAP_URGENT + 2 * LSB   # -2.0 is past ACCEL_URGENT
      prev, step_prev = a, step

  def test_gas_override_drops_to_inactive_at_once_and_restarts_from_measured(self):
    controller, cs = make_long_car()
    cs.a_ego_wheel = 0.0
    for _ in range(50):
      tick(controller, cs, 1.0)
    assert 0.0 < controller.accel_last < 49 * UP   # first frame measured (0.0), then an eased rise
    assert tick(controller, cs, 1.0, gas=True) is None   # OEM's own frame (its override handling) goes through
    assert controller.accel_last == 0.0
    cs.a_ego_wheel = 0.6
    req = tick(controller, cs, -1.0)
    assert abs(req["ADAS_LgtCtrl_AccelReq"] - 0.6) < LSB

  def test_disengage_is_inactive_at_once(self):
    controller, cs = make_long_car()
    for _ in range(10):
      tick(controller, cs, -3.0)
    assert tick(controller, cs, -3.0, long_active=False) is None
    assert controller.accel_last == 0.0

  def test_standstill_hold_frames(self):
    # openpilot's LongControl holds CarParams.stopAccel (default -2.0) once stopped. What goes out: a plain ACC
    # decel request in OEM's 0x121; standstill / hold stay the OEM ADAS's (we send no 0x117 / 0x118 at all).
    controller, cs = make_long_car()
    cs.out.vEgoRaw = 0.0
    cs.out.standstill = True
    for _ in range(200):
      req = tick(controller, cs, -2.0)
    assert abs(req["ADAS_LgtCtrl_AccelReq"] + 2.0) < LSB
    assert req["ADAS_LgtCtrl_AccelVld"] == 1

  def test_new_actuators_report_the_sent_accel(self):
    controller, cs = make_long_car()
    cs.a_ego_wheel = 0.2
    cc = structs.CarControl(longActive=True)
    cc.actuators.accel = -1.0
    cs.out.cruiseState.enabled = True
    new_actuators, _ = controller.update(cc.as_reader(), structs.CarControlSP(), cs, 0)
    assert abs(new_actuators.accel - 0.2) < 1e-6
