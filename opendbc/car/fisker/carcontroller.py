import math
import numpy as np

from opendbc.can import CANPacker
from opendbc.car import DT_CTRL, Bus, structs
from opendbc.car.carlog import carlog
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.lateral import apply_std_steer_angle_limits

from opendbc.car.fisker.fiskercan import FiskerCAN
from opendbc.car.fisker.secoc import stamp_secoc, stamp_secoc_many, sync_mac
from opendbc.car.fisker.values import (ANGLE_OVERRIDE_DEG, ANGLE_OVERRIDE_RELEASE_DEG, OVERRIDE_STEER_TORQUE,
                                        TURN_COOP_MAX_SPEED, TURN_COOP_UNWIND_DEG,
                                        CarControllerParams, ICC_0x35B_OVERRIDES,
                                        SETTLE_ANGLE_STEP, SETTLE_MAX_TICKS, STANDSTILL_PAUSE_SPEED)
from opendbc.sunnypilot.car.fisker.values_ext import FiskerFlagsSP


VisualAlert = structs.CarControl.HUDControl.VisualAlert

# CAN IDs of the SecOC-protected actuator messages we transmit.
STEER_CAN_ID = 0x1D0
ACCEL_CAN_ID = 0x121

# Phone-over-wifi remote-control gating. See
# openpilot/sunnypilot/selfdrive/remote_control/remote_control_server.py for
# the Params producer. All three are CLEAR_ON_MANAGER_START so a reboot
# always comes up with the app disabled. RemoteControlEnabled is the owner's
# opt-in: manager only runs the server while it is set, and the server only
# writes a fresh angle after a client has armed it.
#   * stale after this long: carcontroller falls back to sunnypilot's planner.
#     Chosen to be ~10× the server's command interval (~33 ms @ 30 Hz) so a
#     single dropped packet is survivable but a dead wifi link disengages
#     within one 1/3 second window.
REMOTE_STALE_NS = 300_000_000  # 300 ms
#   * low-speed-only. 10 mph in m/s. Above this we refuse to apply the remote
#     angle no matter what the server says — a wifi-latency-limited slider is
#     not an acceptable highway input device.
REMOTE_MAX_SPEED_MS = 10 * 0.44704  # ~4.47 m/s
#   * poll the params lazily — opendbc test harness on macOS doesn't have
#     libparams_c and Params() crashes test_car_interfaces at import. On
#     device the import succeeds. Follows the same shape as
#     carstate._set_mads_arm above.
#   * keep param I/O OUT of the 100 Hz control hot path. A Params.get() is a
#     filesystem read; doing one (or three) every tick is exactly the kind of
#     jitter that can stall the lateral TX loop and leave a gap the EPS rejects
#     (card.py deliberately reads its own params at 10 Hz in a side thread for
#     the same reason). We re-read the RemoteControlEnabled gate only every
#     REMOTE_POLL_FRAMES ticks; when it's off (every normal drive) the hot path
#     does zero I/O between polls. When it's on we read the live angle every
#     tick — that's the active parking feature and must stay responsive.
REMOTE_POLL_FRAMES = 10  # re-read the enable gate at ~10 Hz (100 Hz loop)

# 0x1D0 SecOC candidate pool (race-free handover; see opendbc/safety/modes/fisker.h).
# Instead of transmitting one 0x1D0 on bus 0 and fighting OEM's identical next frame for
# the EPS, we ship a small WINDOW of fully-stamped 0x1D0 payloads each tick as 0x5D0
# addressed to bus 1. Panda buffers them and substitutes the one matching OEM's OWN current
# counter in place of OEM's bus-2 0x1D0 — so the EPS rides OEM's counter with our angle,
# with no duplicate/backward value at engage or disengage. The window straddles OEM's
# counter as carstate last saw it, covering the Python→boardd→USB latency so the value
# panda needs is always present; a miss is just a dropped frame, never a bad counter.
# Bus 1 is the MRR's private link (receive only), so panda's safety buffers each
# candidate and then rejects it: nothing is ever transmitted there.
LAT_CANDIDATE_ADDR = 0x5D0   # carries an 0x1D0 payload; panda-internal (never on a real bus)
LONG_CANDIDATE_ADDR = 0x521  # carries an 0x121 payload; panda-internal, same bus and handling as 0x5D0
LAT_CANDIDATE_BUS = 1        # addressed to the MRR link; panda rejects it after buffering
# The window has to cover the whole round trip: OEM frame -> panda -> pandad -> card ->
# sendcan -> panda. Candidate k is used ~(10*k - latency) ms after it reaches panda, and
# panda only uses candidates younger than FISKER_CAND_FRESH_US (60 ms). Upstream's -1..+3
# window covers ~30 ms. On a comma 3X running StarPilot that wasn't enough: almost every
# substitution missed, the EPS saw 0x1D0 gaps and aborted with AbortFb=8 ~100 ms after going
# Active. -1..+9 covers round trips up to ~90 ms; later candidates for the same counter
# overwrite earlier ones in panda, so the angle used is always the newest available.
LAT_CANDIDATE_WINDOW = 11    # number of candidates per tick
LAT_CANDIDATE_LEAD = 1       # how many below OEM's observed counter to start (straddle it)

# GW reset boundary. About once a second the GW reset counter steps and OEM restarts its
# 0x1D0 SecOC counter at 1. The window above follows OEM's counter as carstate last saw it,
# so until carstate sees the restart nothing matches and panda drops OEM's frames: on
# 2026-10-28 that was 5 frames in a row (a 20 ms card stall on top of the ~2-frame
# pipeline lag) and the EPS aborted (AbortFb=8 -> LKAS fault). So from late in each window
# we also pre-stamp the first counters of the NEXT window (reset + 1). OEM's alive counter
# runs straight through the reset, so the alive those frames need depends on which counter
# the current window ends on, 100, 101 or 102 on that drive: we send one variant per
# possible end and panda emits the one whose alive matches OEM's frame. (The earlier
# post-reset set, one guessed alive per counter for 4 ticks after the GW_Syn_All edge,
# replayed against that drive gave ~3 wrong-alive frames per reset: once carstate caught up,
# its guesses overwrote the window's correct candidates in panda.)
LAT_NEXT_WINDOW_FROM = 94           # start pre-stamping once OEM's counter is this far in
LAT_NEXT_WINDOW_COUNT = 6           # first counters of the next window to cover (60 ms)
LAT_WINDOW_ENDS = (100, 101, 102)   # possible last counters of a window (panda keeps 4 per counter)


def oem_reset_value(gw_reset, oem_flag):
  """The full GW reset value of OEM's current window. GW_Syn_All steps ~10 ms before OEM
  restarts its counter, so OEM's window is the GW's, or the one before it."""
  for reset in (gw_reset, gw_reset - 1, gw_reset + 1):
    if (reset & 0x3) == oem_flag:
      return reset
  return gw_reset


def lat_candidates(oem_full, oem_alive, oem_reset):
  """(secoc counter, alive, reset) for each 0x1D0 candidate to stamp this tick. Each one is
  a frame some number of frames from OEM's current one; counter and alive both step +1 per
  frame, so the pair stays self-consistent (the MAC covers alive via byte 1)."""
  cands = []
  for k in range(-LAT_CANDIDATE_LEAD, LAT_CANDIDATE_WINDOW - LAT_CANDIDATE_LEAD):
    if oem_full + k < 1:
      continue  # counter restarts at 1 each window; <1 is never what the EPS expects
    cands.append((oem_full + k, (oem_alive + k) % 15, oem_reset))
  if oem_full >= LAT_NEXT_WINDOW_FROM:
    for end in LAT_WINDOW_ENDS:
      if end < oem_full:
        continue  # OEM is already past this end, so its window didn't end there
      for ctr in range(1, LAT_NEXT_WINDOW_COUNT + 1):
        cands.append((ctr, (oem_alive + end - oem_full + ctr) % 15, oem_reset + 1))
  return cands


def apply_fisker_accel_limits(target, last, measured, active, was_active):
  """The 0x121 accel request for this frame.

  Not active (not long-active, or the driver is on the gas): the inactive value 0.0, at once.
  First active frame: the measured acceleration, so the request does not jump at engage.
  After that: move toward the target by at most ACCEL_JERK_UP / ACCEL_JERK_DOWN per frame.
  The result is always within ACCEL_MIN..ACCEL_MAX."""
  P = CarControllerParams
  if not active:
    return 0.0
  if not was_active:
    return float(np.clip(measured, P.ACCEL_MIN, P.ACCEL_MAX))
  target = float(np.clip(target, P.ACCEL_MIN, P.ACCEL_MAX))
  step = float(np.clip(target, last - P.ACCEL_JERK_DOWN * DT_CTRL, last + P.ACCEL_JERK_UP * DT_CTRL))
  return float(np.clip(step, P.ACCEL_MIN, P.ACCEL_MAX))


def coop_signaled_turn(desired_angle, wheel_angle, wheel_step, v_ego, left_blinker, right_blinker):
  """True when taking the wheel would only continue the driver's own signaled low-speed turn: the blinker is on
  toward the side the wheel is turned, the driver is turning into it or holding (not unwinding: wheel_step, the
  wheel's change since the last tick, is not moving it back toward center faster than TURN_COOP_UNWIND_DEG per
  tick), and the command is on that side too and wants at least as much turn (within ANGLE_OVERRIDE_RELEASE_DEG)
  as the wheel has. A driver straightening out of the turn is never "cooperating". Angles in deg, positive = left."""
  if v_ego >= TURN_COOP_MAX_SPEED or desired_angle * wheel_angle <= 0.0:
    return False
  if wheel_step * math.copysign(1.0, wheel_angle) < -TURN_COOP_UNWIND_DEG:
    return False
  toward = (left_blinker and not right_blinker and wheel_angle > 0.0) or (right_blinker and not left_blinker and wheel_angle < 0.0)
  return toward and abs(desired_angle) >= abs(wheel_angle) - ANGLE_OVERRIDE_RELEASE_DEG


def apply_fisker_steer_angle_limits(desired_angle, last_angle, speed, measured_angle, lat_active, controls_permitted):
  # Req=0 releases the EPS, but Panda still rate-checks 0x1D0 while ACC/MADS permits
  # control. Follow the driver's measured angle at the normal rate instead of
  # snapping to it and dropping frames at Panda. With both permissions off, Panda
  # requires a measured-angle heartbeat instead.
  target = desired_angle if lat_active else measured_angle
  return apply_std_steer_angle_limits(target, last_angle, speed, measured_angle,
                                     controls_permitted, CarControllerParams.ANGLE_LIMITS)


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP, CP_SP):
    super().__init__(dbc_names, CP, CP_SP)
    self.params = CarControllerParams
    self.packer = CANPacker(dbc_names[Bus.pt])
    self.fcan = FiskerCAN(CP, self.packer)

    self.apply_angle_last = 0.0

    # 0x121 accel request after the jerk limiter (apply_fisker_accel_limits), and whether the last
    # frame was actively commanding (the limiter restarts from the measured accel when it was not).
    self.accel_last = 0.0
    self.was_accel_active_prev = False

    # Longitudinal sends only 0x121 candidates, stamped with OEM 0x121's own alive / SecOC counters (lat_candidates).
    self.was_tx_long_prev = False   # 0x121 candidates sent last tick

    self.secoc_key_verified = False
    self.secoc_warn_logged = False

    # Once openpilot has been long-active at least once, keep the ADAS heartbeat going for
    # the rest of the drive. If we go silent between engagements the ESP loses our 0x118 AEB
    # state (falls into "AEB unavailable" fault) and the VCU's E2E AliveCounter validator
    # rejects our first re-engage frames as out-of-sequence — surfacing as "ADAS error,
    # emergency brake unavailable" on the cluster and a cruise fault in openpilot.
    self.long_ever_active = False

    # "We ARE OEM" AliveCounter substitution: every OEM 0x1D0/0x1C0 tick that panda
    # blocks, we transmit a frame with the SAME AliveCounter that OEM would have sent —
    # our stream literally continues OEM's numbering during the blocked window. At
    # disengage panda unblocks OEM, and OEM's next tick lands at our_last + 1 which is
    # exactly what EPS was expecting next — seamless handoff both directions, no
    # momentary BSM error on disengage (see user's diagram). Only transmit when OEM's
    # bus-2 counter has advanced since our previous send, else we'd emit a duplicate
    # (EPS rejects duplicates in the E2E monotonic check).
    self.alive_1d0 = 0
    self.alive_1c0 = 0
    self.was_lat_engaged_prev = False   # tracks lat_engaged (MADS) rising edge for engage delay
    self.was_tx_lateral_prev = False    # rising edge of lateral TX -> snap alive counters from OEM
    self._override_eps_recovery = False # EPS latched Off by a driver overpower -> de-assert to reset
    self._eps_was_active = False        # EPS reached Active this engagement (gates override recovery)
    self.was_lat_active_prev = False    # rising edge of lat_active within a session -> smooth re-entry
    self._settle_hold = 0               # ticks left to wait for the wheel to stop self-centering
    self._prev_steer_angle = 0.0        # for the settle angle-rate check
    self._angle_override_latch = False  # measured wheel diverged far from command -> driver steering
    self._coop_release = False          # the latch was just released by a cooperative signaled turn

    # ICC 0x52A spoof: emit ONE frame per OEM tick (detected by AliveCounter change).
    # ICC broadcasts at ~10 Hz while we control at 100 Hz, so a naive tx-every-tick
    # would emit ~10 duplicates per OEM tick — the receiver's E2E monotonic-counter
    # validator would flag those as replays and reject them.
    self.last_spoofed_icc_alive = -1
    # EPS_0x1C2 spoof: same cadence rule (one spoof per real OEM tick, ~50 Hz).
    self.last_spoofed_eps_1c2_alive = -1

    # Lateral fade-out: ticks remaining of post-disengage grace. On MADS falling edge
    # we extend the TX window by this many ticks so the EPS sees Req=0 for a beat
    # and transitions Active→Available smoothly, before our stream fully drops. Then
    # the panda lateral relay (60 ms) lets OEM resume to the EPS right after our last
    # fade frame — our counter is OEM's exact continuation so OEM picks up at our_last+N
    # with a tiny N (clean handoff). Keep the fade short: 10 ticks @ 100 Hz = 100 ms is
    # plenty for the EPS to see Req=0 and release, and keeps the total post-disengage
    # blackout (fade + relay) to ~160 ms instead of the old ~1 s. Was 50 (500 ms).
    self.lat_fade_remaining = 0
    self.FADE_TICKS = 10

    # Engagement delay: wait this many ticks after lat_engaged rising edge
    # before we start TX'ing. Between our carstate's OEM-counter snapshot
    # and our first TX arriving at EPS, OEM can send one more 0x1D0 frame
    # on bus 2 that panda forwards to bus 0 (panda's lateral relay only
    # activates after our FIRST TX). If that OEM frame beats our TX to EPS,
    # EPS's last-accepted counter == our TX counter → duplicate → abort=8.
    # We re-snap each tick during the delay to track OEM's latest wire value.
    #
    # The delay ALSO covers a second, larger race seen on-vehicle 2026-10-26 (in the
    # StarPilot port, on a no-ACC-coded car): the panda's lateral authorization
    # (there Always On Lateral's aol_allowed; here MADS's controls_allowed_lateral)
    # rose ~180 ms AFTER openpilot first decided lat was engaged,
    # because the two read the vehicle cruise/MFS state through independent code
    # paths (software carstate vs. safety firmware) with different latencies.
    # While openpilot TXes candidates but the panda is not yet authorized, panda
    # arms the relay (blocking OEM's 0x1D0/0x1C0 from the EPS) but won't substitute
    # our candidate → the EPS receives NEITHER OEM nor our frames → it starves and
    # aborts (abort=8, EPS Available→Active never completes). The moving-engage log
    # showed exactly this: a ~160 ms burst of CA=False drops right at the
    # Available→Active transition. Holding off our TX until the delay expires keeps
    # the relay disarmed (OEM keeps feeding the EPS) across the whole authorization
    # window, so we only ever start injecting once the panda can actually deliver.
    # 25 ticks (250 ms @ 100 Hz) covers the measured ~180 ms race with margin while
    # keeping the engage handoff short; the OEM heartbeat steers the car during the
    # hold (OEM ADAS is in Standby on this car, so it's just the keep-alive stream).
    self.engage_delay_remaining = 0
    self.ENGAGE_DELAY_TICKS = 25

    # Phone-over-wifi remote-control state. Lazy-init'd on first update() tick
    # so the opendbc test harness (which has no libparams_c on macOS) can
    # instantiate CarController without blowing up. See REMOTE_* constants.
    self._remote_params = None
    self._remote_warned = False
    self._remote_enabled = False     # cached RemoteControlEnabled gate (polled at ~10 Hz)
    self._remote_poll_frame = -REMOTE_POLL_FRAMES  # force a poll on the first tick
    self._remote_active = False      # the phone's angle is the one commanded this tick

  def _maybe_verify_key(self, CS) -> None:
    """Verify the stored SecOC key against the GW sync MAC once at startup."""
    if self.secoc_key_verified or not CS.secoc_sync_seen or not self.secoc_key:
      return
    if sync_mac(self.secoc_key, CS.secoc_trip, CS.secoc_reset) == CS.secoc_sync_mac:
      self.secoc_key_verified = True
      carlog.info("Fisker SecOC key verified against GW sync MAC")
    elif not self.secoc_warn_logged:
      carlog.error("Fisker SecOC key mismatch — GW sync MAC does not match stored SecOCKey")
      self.secoc_warn_logged = True

  def _stamp(self, msg, can_id, trip, reset, msg_counter):
    """Fill the SecOC tail (fresh byte + MAC) of a packed frame with the relay's SecOC counter."""
    addr, data, bus = msg
    stamped = stamp_secoc(self.secoc_key, can_id, data, trip, reset, msg_counter)
    return addr, stamped, bus

  def _remote_steer_override(self, planner_angle_deg: float, v_ego: float, now_nanos: int) -> float:
    """If the phone remote-control app is armed AND fresh AND we're below the low-speed
    cap, replace the lateral planner's commanded steering angle with the phone's angle.
    All downstream safety (rate limiter, driver intervention, SecOC, panda tx_hook)
    runs on top of the substituted value unchanged — same as if sunnypilot itself had
    commanded that angle.

    Returns the (possibly substituted) commanded angle. Fails open: any error
    (Params unavailable, parse failure, stale timestamp, overspeed) falls back to the
    planner angle so sunnypilot continues to control steering normally."""
    self._remote_active = False
    if self._remote_params is None:
      try:
        from openpilot.common.params import Params
        self._remote_params = Params()
      except Exception:  # no params -> never override
        return planner_angle_deg

    # Re-read the enable gate only every REMOTE_POLL_FRAMES ticks (~10 Hz) to keep
    # the filesystem read out of the 100 Hz hot path. Between polls we trust the
    # cached value; a ~100 ms latency to notice arm/disarm is fine (the staleness
    # watchdog below, which runs every tick while armed, is the real safety net).
    if self.frame - self._remote_poll_frame >= REMOTE_POLL_FRAMES:
      self._remote_poll_frame = self.frame
      try:
        self._remote_enabled = self._remote_params.get_bool("RemoteControlEnabled")
      except Exception:  # any error → treat as disarmed
        self._remote_enabled = False
    if not self._remote_enabled:
      return planner_angle_deg

    # Armed: read the live angle every tick so the parking feature stays responsive.
    try:
      last_raw = self._remote_params.get("RemoteSteerLastUpdate")
      angle_raw = self._remote_params.get("RemoteSteerAngle")
      if not last_raw or not angle_raw:
        return planner_angle_deg
      last_ns = int(last_raw.decode() if isinstance(last_raw, (bytes, bytearray)) else last_raw)
      remote_angle = float(angle_raw.decode() if isinstance(angle_raw, (bytes, bytearray)) else angle_raw)
    except Exception:  # any corruption -> fall back to planner
      return planner_angle_deg

    # Freshness watchdog. The phone POSTs /steer at ~30 Hz; a connection drop
    # or wifi stall shows up here as the age blowing past REMOTE_STALE_NS and
    # we silently hand steering back to sunnypilot's planner. The server-side
    # watchdog on the carcontroller is the authoritative one — the server
    # also auto-disarms if the UI loses connectivity, but we don't rely on
    # that because the carcontroller may still have a stale "armed=True" with
    # a last-update timestamp hours old.
    if now_nanos - last_ns > REMOTE_STALE_NS:
      return planner_angle_deg
    # Low-speed only. The whole point of this feature is to steer the car
    # at parking speeds from outside the vehicle; a wifi-latency-limited
    # slider is not a replacement for an ADAS at any speed above walking pace.
    if v_ego > REMOTE_MAX_SPEED_MS:
      return planner_angle_deg
    self._remote_active = True
    return remote_angle

  def update(self, CC, CC_SP, CS, now_nanos):
    actuators = CC.actuators
    can_sends = []

    self._maybe_verify_key(CS)
    secoc_ok = self.CP.secOcKeyAvailable and self.secoc_key_verified
    trip, reset = CS.secoc_trip, CS.secoc_reset

    # Alive counters are seeded-once-at-block then free-run — lateral right before the
    # lateral TX below. Longitudinal 0x121 candidates use OEM 0x121's counters directly (the long section).

    # E2E AliveCounter (byte1 low nibble): 0..14 counter, +1 per frame, never 15 (15 is
    # the E2E invalid sentinel — DBC range [0|14]). Every PDU we send (lateral 0x1D0/0x1C0,
    # longitudinal 0x121/0x117/0x118) tracks its own counter so we can align to OEM's most
    # recent bus-2 value at every TX start — that way our first frame is (OEM_last + 1),
    # exactly what the receiver expects. If we free-ran with self.frame%15 through idle
    # periods, the receiver's last-accepted-from-OEM would diverge from our next-transmit and
    # the first frame would be rejected (surfaces as an LKA fault / "ADAS error" on cluster
    # and MADS auto-off).

    # ---- Lateral (steering angle 0x1D0 + activation 0x1C0 @ 100 Hz) ----
    # Lateral TX is driven by MADS, NOT by cruise. In ACC-only mode (RiBtnNorth engaged
    # ACC without arming MADS) openpilot stays transparent on the lateral path so the
    # OEM ADAS keeps running its own LKA (gated by the user's ICC_LKASetting). The
    # panda's fisker_fwd_hook has the matching rule: it only blocks OEM 0x1D0/0x1C0 on
    # `controls_allowed_lateral`, not plain `controls_allowed` — otherwise the OEM
    # thinks LKA has become unavailable because its 0x1D0 never reaches bus 0.
    #
    # Engagement variables:
    #   `engaged`     – ACC OR MADS. Gates the long path + SecOC window-counter housekeeping.
    #   `lat_engaged` – MADS only. Gates the lateral TX (0x1D0 + 0x1C0) and the
    #                   OEM-counter seeding (which only matters when we're about to start
    #                   transmitting). Rising edge of lat_engaged is the moment we snap
    #                   our aliveCounter / SecOC wire counter to OEM's current value so
    #                   our first TX continues OEM's numbering seamlessly.
    #
    # Driver-override release. CS.driver_intervening is driven by the EPS's dedicated
    # EPS_DrvrIntvSteerWhlDetd bit (0x1C4) — a conservative "real driver overpower"
    # signal that EPS computes with internal knowledge of its motor contribution, so
    # it doesn't false-fire from reaction torque the way a plain column-torque
    # threshold does. When it trips we drop lat_active so ADAS_LatCtrl_Req=0 and the
    # EPS stops servoing. We intentionally do NOT key this off CS.out.steeringPressed
    # — that's the sensitive, column-torque-based "driver is touching the wheel"
    # signal, and consuming it here caused the EPS to release on every light nudge
    # (which also killed sunnypilot's auto-lane-change nudge detection since ALC
    # watches steeringPressed too). Separating the two sensitivities lets light
    # nudges feed ALC while hard overpower releases the EPS.
    # Use mads.active (= state ∈ {enabled, softDisabling, overriding}) rather than
    # mads.enabled, which also returns True for State.paused. MADS can enter paused
    # directly from disabled (gear/door/seatbelt NO_ENTRY condition + silent LKAS
    # disable), which has no engagement event — so keying off `enabled` makes
    # lat_engaged latch True indefinitely from startup and we TX zeros at 100 Hz
    # even in pure ACC-only. The EPS then sees our Sts=0 stream racing OEM's own
    # lateral once ACC becomes Active and raises abort=8 (CAN).
    #
    # mads.active only lights up when MADS is actually commanding steering, which
    # is the only time we have any business TXing on 0x1D0/0x1C0.
    mads_active = bool(CC_SP.mads.active) if CC_SP is not None else False
    engaged = (CS.out.cruiseState.enabled or mads_active) and secoc_ok
    lat_engaged = mads_active and secoc_ok
    # Phone remote-control substitution: swap the planner's commanded angle for
    # the phone-supplied angle BEFORE the rate limiter. Keeping it before the
    # rate limiter means the exact same slew/saturation limits that constrain
    # the sunnypilot planner also constrain the phone — the phone can't send a
    # ±600° step command and bypass the EPS's safety envelope. All of our other
    # safety (SecOC, driver intervention, panda tx_hook angle rate limits) is
    # downstream of this substitution and runs unchanged.
    #
    # Gated by lat_engaged: MADS still has to be commanding steering for the
    # phone's commands to reach the EPS. If MADS isn't engaged, we're not
    # transmitting 0x1D0 anyway and the substituted actuators.steeringAngleDeg
    # just affects what new_actuators carries back on the car bus for logging.
    desired_angle_deg = float(actuators.steeringAngleDeg)
    self._remote_active = False
    if lat_engaged:
      desired_angle_deg = self._remote_steer_override(desired_angle_deg,
                                                       CS.out.vEgoRaw, now_nanos)

    # Pause actuation below ~1 mph so the wheel doesn't creep/center at a stop (we keep the
    # session and feed the EPS with Req=0, same as a driver-override pause).
    # Not while the phone is steering: remote control is for parking, at walking pace or stopped.
    standstill_pause = CS.out.vEgoRaw < STANDSTILL_PAUSE_SPEED and not self._remote_active
    # Angle-divergence override: measured wheel far from the commanded angle (the model's, or the
    # phone's while it steers) => the driver is steering independently (hard manual input the
    # torque yield missed because the EPS was assisting). Yield instead of commanding into it.
    # Latch with hysteresis so it holds through the maneuver and clears once the wheel is back
    # near the command.
    # Trip only when the wheel is far from BOTH the model's target and what we actually commanded last tick
    # (apply_angle_last, rate-limited). In a sharp turn the model's target runs well ahead of the rate-limited
    # command and the wheel follows the command; that is us steering, not the driver (on-vehicle 2026-11-14, four
    # signaled low-speed turns handed back mid-turn). A driver turning past our command still trips. Release stays
    # on the target: while latched apply_angle_last is glued to the wheel, so it can't tell when the driver is done.
    angle_diverge = abs(CS.out.steeringAngleDeg - desired_angle_deg)
    command_diverge = abs(CS.out.steeringAngleDeg - self.apply_angle_last)
    # Cooperative signaled turn: the driver started a signaled low-speed turn ahead of us and the command now wants
    # at least as much turn, the same way, as the wheel has. Taking over then just carries on the driver's turn, so
    # it is not an override (on-vehicle 2026-11-14: the driver started every signaled turn 0.2-1.8 s before the
    # turn planner, the latch tripped, and since the planner's target never came back within RELEASE_DEG of a wheel
    # held through the turn, it handed every turn back for good). Pushing against us still trips the torque override.
    coop_turn = coop_signaled_turn(desired_angle_deg, CS.out.steeringAngleDeg,
                                   CS.out.steeringAngleDeg - self._prev_steer_angle, CS.out.vEgoRaw,
                                   CS.out.leftBlinker, CS.out.rightBlinker)
    if angle_diverge > ANGLE_OVERRIDE_DEG and command_diverge > ANGLE_OVERRIDE_DEG and not coop_turn:
      self._angle_override_latch = True
    elif angle_diverge < ANGLE_OVERRIDE_RELEASE_DEG or coop_turn:
      self._coop_release = self._coop_release or (self._angle_override_latch and coop_turn)
      self._angle_override_latch = False
    # The same cooperative turn sets our torque latch aside (never the EPS's own overpower bit): a driver who pushed
    # the wheel into the turn trips it, and holding a turned wheel against its self-centering takes 1.6-2.6 Nm, so
    # it never falls below OVERRIDE_RELEASE_TORQUE until the turn is over (on-vehicle 2026-11-14, every signaled turn).
    # Only while the driver's torque is below the trip level and holds into the turn: pulling back toward center,
    # however lightly, is an override.
    driver_intervening = CS.driver_intervening
    if (driver_intervening and coop_turn and getattr(CS, "torque_latch_only", False)
        and abs(CS.out.steeringTorque) < OVERRIDE_STEER_TORQUE and CS.out.steeringTorque * CS.out.steeringAngleDeg >= 0.0):
      driver_intervening = False
      self._coop_release = True
    lat_active = (CC.latActive and secoc_ok and not driver_intervening
                  and not standstill_pause and not self._angle_override_latch)

    # Smooth re-entry after any pause (driver override release, or crossing the standstill
    # speed) while the session stays engaged: re-arm the engage delay so we don't snap. During
    # the hold apply_angle_last re-seeds to the live wheel angle (see handoff seeding below)
    # and the counters re-snap, so we pick up from where the wheel actually is and slew to the
    # target rather than yanking it. The fresh-engage rising edge (below) handles the very
    # first engage; this handles resumes within a session.
    resume_edge = lat_active and not self.was_lat_active_prev and self.was_lat_engaged_prev
    if resume_edge:
      self.engage_delay_remaining = self.ENGAGE_DELAY_TICKS
      # Mid-turn the driver is still turning the wheel, so waiting for it to settle would hold off up to 1.5 s,
      # most of the turn. We pick up from the wheel and slew the same way it is already going.
      self._settle_hold = 0 if self._coop_release else SETTLE_MAX_TICKS
    if lat_active or not CC.latActive:
      self._coop_release = False
    # Settle wait: a fixed hold still grabbed a wheel that was still self-centering fast on
    # release, which fought the wheel — a jerk, a transient EPS abort ("steering temporarily
    # unavailable"), and the rate limiter saturating (a false "limit" alert). On-vehicle
    # 2026-10-26. So while settling, keep the hold extended as long as the wheel is still
    # moving (> SETTLE_ANGLE_STEP deg/tick), and only let the engage delay run out and re-grab
    # once it has steadied. Bounded by SETTLE_MAX_TICKS so it can't wait forever.
    angle = CS.out.steeringAngleDeg
    if self._settle_hold > 0:
      self._settle_hold -= 1
      if abs(angle - self._prev_steer_angle) > SETTLE_ANGLE_STEP:
        self.engage_delay_remaining = self.ENGAGE_DELAY_TICKS
      else:
        self._settle_hold = 0
    self._prev_steer_angle = angle
    self.was_lat_active_prev = lat_active

    # Rate limiting through the handover (see apply_fisker_steer_angle_limits): on a
    # driver override or in the post-disengage fade, Req drops to 0 but 0x1D0 keeps
    # slewing to the measured angle at the normal rate while ACC or MADS still permits
    # control, so Panda's rate check keeps accepting the frames; with neither permission
    # it carries the measured angle as the heartbeat.
    self.apply_angle_last = apply_fisker_steer_angle_limits(
      desired_angle_deg, self.apply_angle_last, CS.out.vEgoRaw,
      CS.out.steeringAngleDeg, lat_active, engaged,
    )

    # "We ARE OEM" counter continuation. The lateral counters (alive_1c0, and the 0x1D0
    # candidates' alive + SecOC counter) come from OEM's authoritative bus-2 values + 1. OEM's counters advance in real time regardless of
    # our control-loop jitter, so mirroring them keeps our injected stream on exactly
    # the value the EPS expects next, both at the engage handoff and on every frame
    # after. Panda blocks OEM's own 0x1D0/0x1C0 from the EPS while we TX, so the EPS
    # sees a seamless continuation of OEM's numbering through our frames.
    #
    #   OEM on bus 2 (all ticks):        1  2  3  4  5  6  7  8  9  10 11 12
    #   Panda forwards to bus 0:         1  2  3  4                    11 12
    #                                    └── ENGAGE ──┘     └─ DISENGAGE
    #   Our TX (= OEM_bus2 + 1):                   6  7  8  9  10 11
    if lat_engaged and not self.was_lat_engaged_prev:
      # Rising edge: start the engage-delay window. During the delay we DO NOT TX —
      # we let OEM's own 0x1D0/0x1C0 keep flowing to the EPS so the handoff point is
      # clean, and our counters are already mirroring OEM continuously (set at the
      # top of update()), so the first TX after the delay lands on OEM_current + 1.
      self.engage_delay_remaining = self.ENGAGE_DELAY_TICKS
    if lat_engaged and self.engage_delay_remaining > 0:
      self.engage_delay_remaining -= 1
    # Fade-out window: on lat_engaged falling edge we extend the TX window by FADE_TICKS
    # so the EPS keeps receiving our Req=0 stream for a beat after disengage. During
    # fade: panda's time-based lateral relay keeps blocking OEM 0x1D0/0x1C0, the EPS
    # 0x1C2 spoof keeps running (OEM ADAS still sees EPS=Available), and the EPS
    # transitions Active→Available smoothly off our commands rather than off a sudden
    # OEM-command injection. The 0x1C0 wire fields during fade keep Sts=1/Typ=1 (we
    # haven't "left") and only drop Req to 0 — observed on-vehicle 2026-10-18 that
    # slamming Sts/Typ to 0 at disengage caused EPS abort=8 the same way the mid-
    # engage driver-override transition did.
    # Fade is PRIMED (reset to FADE_TICKS) on every engaged tick so that the
    # moment we disengage it's already full. It's CONSUMED (counted down)
    # only when not engaged. The subtle part: `tx_lateral` must only treat
    # the fade as "active" during the post-disengage window — never during
    # engagement — otherwise priming the fade on the rising edge immediately
    # satisfies the TX gate and we bypass the engage delay. That's exactly
    # what showed up on-vehicle 2026-10-19: first TX was happening ~15 ms
    # into engagement (one tick) instead of ~50 ms (after the 5-tick delay),
    # which bypassed the OEM-counter settle and let the first few SecOC
    # frames go out with a counter the EPS hadn't caught up to yet.
    if lat_engaged:
      self.lat_fade_remaining = self.FADE_TICKS
      in_fade = False
    else:
      in_fade = self.lat_fade_remaining > 0
      if in_fade:
        self.lat_fade_remaining -= 1
    self.was_lat_engaged_prev = lat_engaged
    tx_lateral = (lat_engaged and self.engage_delay_remaining == 0) or in_fade

    # EPS override-latch recovery. On-vehicle 2026-10-26: a hard overpower drives the EPS to
    # Off (EPS_AdasLatCtrlSts == 0) and it will NOT come back while we keep asserting
    # engagement (Sts=1/Req=0). It sat at sts=0 / intervention=1 for ~48 s — through the
    # driver repeatedly releasing — until openpilot stopped asserting, then recovered to
    # Available in ~0.1 s. So the EPS needs us to DE-ASSERT (stop transmitting -> Sts=0) to
    # clear the override, then a fresh engage to return. Previously only a MADS off/on cycle
    # supplied that, so lateral paused on the first overpower and never resumed on its own.
    #
    # While engaged and the EPS reads Off, stop transmitting so OEM's own 0x1D0/0x1C0 flow to
    # the EPS and it resets to Available. Once it's back (sts != 0) AND the driver has let go
    # (not steeringPressed), re-arm the normal engage delay: OEM keeps feeding the EPS through
    # the hold, the SecOC/alive counters re-snap from OEM on the first TX after (was_tx_* went
    # False while we were de-asserted), and apply_angle_last re-seeds to the measured wheel
    # angle — a clean fresh handshake, so steering resumes with no button press.
    # Gate recovery on the EPS having actually come up this engagement: only a drop to Off
    # AFTER it reached Active is an overpower knock-out. At initial engage the EPS may read
    # Off/Available before our handshake brings it up — we must keep transmitting then, not
    # de-assert, so recovery must not fire until it has been Active at least once.
    if not lat_engaged:
      self._eps_was_active = False
      self._override_eps_recovery = False
    elif CS.eps_lat_sts == 2:
      self._eps_was_active = True
    if lat_engaged and CS.eps_lat_sts == 0 and self._eps_was_active:
      self._override_eps_recovery = True
    if self._override_eps_recovery:
      tx_lateral = False   # de-assert so the EPS can clear its override latch
      if lat_engaged and CS.eps_lat_sts != 0 and not CS.out.steeringPressed:
        self._override_eps_recovery = False
        self._eps_was_active = False
        self.engage_delay_remaining = self.ENGAGE_DELAY_TICKS

    # Driver angle-override: fully de-assert, don't just drop Req. When the wheel has diverged
    # far from our command (driver steering hard, e.g. to full lock) we must STOP transmitting
    # rather than keep our SecOC candidate stream running at an extreme angle — on-vehicle
    # 2026-10-27 a GW SecOC reset landing in that stream at/near full lock dropped frames and
    # aborted (abort=9), latching a permanent LKAS fault on the return-to-center re-grab.
    # De-asserting lets OEM's own 0x1D0 feed the EPS through the maneuver (OEM handles its own
    # resets); the latch clears on return-to-center and the resume-edge re-arm re-grabs cleanly.
    if self._angle_override_latch:
      tx_lateral = False

    # Handoff seeding: whenever we are NOT transmitting lateral (not engaged, or still
    # inside the engage-delay window), glue apply_angle_last to the measured wheel angle.
    # The first frame we send after the hold is then a legal one-step slew FROM the real
    # wheel position, so it sits inside both the panda angle-vs-measured rate check and
    # the EPS's own error window.
    #
    # On-vehicle 2026-10-26 (fisker_handover3.py) the abort burst was exactly this: at a
    # moving engage our command was still near 0 (the model's straight-ahead target slewed
    # in during the delay) while the wheel was already at ~-2.4 deg, so panda dropped every
    # candidate for ~160 ms until the limiter walked the command out to the measured angle —
    # the drops cleared the instant cmd == meas. The counters climbed cleanly through the
    # burst and CA stayed False even on the frames that DID deliver, ruling out SecOC and
    # authorization. Seeding to measured removes the stale reference so there is no slew-in
    # gap. apply_angle_last was computed above by slewing toward the model target; overwrite
    # it here on non-TX ticks (the value is unused on those ticks since we send nothing).
    if not tx_lateral:
      self.apply_angle_last = CS.out.steeringAngleDeg
    elif not self.was_tx_lateral_prev:
      # First frame of a burst: send the measured angle itself, no slew. Panda checks it against
      # its own newest wheel angle, which is a frame or two fresher than carstate's, so a full
      # one-step slew lands on or just past its rate limit. On 2026-11-02 that dropped the first
      # candidate in 3 of 45 re-engages at 47-49 mph; panda then re-based on the wheel angle and
      # rejected every following frame of the slew too, and the EPS aborted (AbortFb=8) ~100 ms
      # later. From here on panda's reference is our own last frame, so full steps pass.
      self.apply_angle_last = CS.out.steeringAngleDeg

    # Lateral E2E alive counters (0x1D0 / 0x1C0): seed once at the first TX, then +1, from
    # each PDU's own OEM source. These are independent 4-bit counters that do NOT restart
    # on a GW Reset, so no reset re-seed — just wrap mod 15.
    if tx_lateral:
      if not self.was_tx_lateral_prev:
        self.alive_1d0 = (int(CS.oem_1d0_alive) + 1) % 15
        self.alive_1c0 = (int(CS.oem_1c0_alive) + 1) % 15
      else:
        self.alive_1d0 = (self.alive_1d0 + 1) % 15
        self.alive_1c0 = (self.alive_1c0 + 1) % 15
    self.was_tx_lateral_prev = tx_lateral

    if tx_lateral:
      # alive_1c0 was seeded/incremented just above; 0x1D0 candidates carry OEM's own counters.
      # Lateral control type: sunnypilot FiskerLateralType toggle. Flag set = LCA/TJA
      # (Typ=3), flag clear = LKA (Typ=1, baseline). The Ocean's EPS has historically
      # only honored LKA, so defaulting to 1 keeps the port working.
      lat_ctrl_typ = 3 if (self.CP_SP.flags & FiskerFlagsSP.LAT_CTRL_LCA.value) else 1
      # Two independent wire bits (see create_lat_control docstring for the full why):
      #   * `lat_engaged_wire` — are we in an engaged lateral session? Drives
      #     Sts/Typ/ReqVld. Stable True across driver-override transients AND
      #     across the fade-out window — only drops once the fade has actually
      #     elapsed and we're about to stop transmitting entirely.
      #   * `lat_active_wire`  — on this tick, are we actively servoing the angle
      #     or backing off for the driver? Drives Req only.
      #
      # Both faults observed on-vehicle 2026-10-18 came from slamming Sts/Typ
      # to 0 while EPS was still in state Active: once on the first driver-
      # torque pulse (carControl.latActive dropped), once on MADS release
      # (lat_engaged dropped and the old fade code TX'd Sts=0/Typ=0). Splitting
      # the bits so Sts/Typ stay 1 while only Req follows lat_active keeps the
      # EPS in Active state right up to the moment we actually stop transmitting.
      lat_engaged_wire = lat_engaged or (self.lat_fade_remaining > 0)
      lat_active_wire = lat_active and lat_engaged
      # 0x1D0: emit a WINDOW of candidates (0x5D0 on bus 1) instead of one direct 0x1D0.
      # Each candidate is a complete 0x1D0 frame stamped for a counter around OEM's CURRENT
      # value — so we continuously mirror OEM (the phase-aliasing that broke continuous-
      # mirror before is now resolved IN PANDA, which picks exactly the counter OEM is on).
      # alive (4-bit, byte 1) and the SecOC full counter advance together from OEM's own
      # values, so every candidate is internally consistent for its counter.
      # Candidates come from lat_candidates(): the window around OEM's counter, plus the next
      # reset window's first counters near the end of this one. Stamp them with the reset of
      # OEM's own window, which lags GW_Syn_All by ~10 ms at each reset.
      oem_reset = oem_reset_value(reset, int(CS.oem_1d0_reset_flag))
      # The angle is the same for every candidate, so pack it once and only redo alive + checksum.
      steer_base = self.fcan.create_steering_control(self.apply_angle_last, 0)
      # All candidates are signed in one AES call (stamp_secoc_many): per-frame signing made card the busiest
      # process on the control core.
      cands = lat_candidates(int(CS.oem_1d0_secoc_full_ctr), int(CS.oem_1d0_alive), oem_reset)
      frames = [(self.fcan.steering_control_with_alive(steer_base, alive_k)[1], trip, reset_k, secoc_k)
                for secoc_k, alive_k, reset_k in cands]
      for stamped in stamp_secoc_many(self.secoc_key, STEER_CAN_ID, frames):
        can_sends.append((LAT_CANDIDATE_ADDR, stamped, LAT_CANDIDATE_BUS))
      can_sends.append(self.fcan.create_lat_control(lat_active_wire, self.alive_1c0,
                                                    driver_override=driver_intervening,
                                                    lat_ctrl_typ=lat_ctrl_typ,
                                                    lat_engaged=lat_engaged_wire))

      # EPS_0x1C2 spoof: DISABLED. On-vehicle testing with the verified checksum
      # algorithm (CRC-8 J1850, data_id=0x90) STILL caused ADAS errors in MADS and
      # Both modes, even though the frames themselves are checksum-valid and the
      # byte-6 override does what it claims. Hypothesis: the OEM ADAS doesn't only
      # look at EPS_AdasLatCtrlSts on 0x1C2 — it probably cross-references with
      # EPS_0x1C4 (driver torque) and/or EPS_0x475 (EPS mode) which we don't spoof,
      # so the "Available" story we're telling on 0x1C2 contradicts the "actively
      # being moved" story the other EPS signals still carry. Spoofing all three
      # consistently would be the next step, but requires brute-forcing more
      # checksums and risks more side-effects on the cluster. The fade-out alone
      # addresses the disengage transient, which is the user-visible symptom.
      #
      # Everything stays in place for easy revival once we have a reason to try
      # again with a multi-signal spoof:
      #   * E2E_PARAMS entry (0x1C2: (0x90, 64)) in fiskercan.py — verified against
      #     200 captured frames (test_checksum.py), can't hurt to keep.
      #   * create_eps_spoof_0x1c2() in fiskercan.py — self-contained helper,
      #     passes its own tests.
      #   * EPS_0x1C2_SIGNALS + the carstate snapshot — same, harmless unused.
      # if self.last_spoofed_eps_1c2_alive != CS.eps_1c2_alive and CS.eps_1c2_values:
      #   can_sends.append(self.fcan.create_eps_spoof_0x1c2(CS.eps_1c2_values))
      #   self.last_spoofed_eps_1c2_alive = CS.eps_1c2_alive

    # ---- Longitudinal: modify-and-forward of OEM's 0x121 ----
    # The OEM ADAS keeps running its own ACC, and its 0x121 / 0x117 / 0x118 always reach the car: we send none of them.
    # Sending our own, built from scratch, made the VCU withdraw VCU_ACCRdy ~50 ms into every engage (route 000000ea,
    # 2026-11-14: 9 of 9). While openpilot longitudinal commands, we ship a window of complete 0x121 frames (our accel,
    # AccelVld=1, OEM's reserved bits, each with OEM's alive and SecOC counter for its slot, stamped) as candidates
    # (0x521, bus 1, panda-internal), exactly like the 0x1D0 candidates. The panda swaps the matching one into OEM's
    # own 0x121 on its way to the VCU, only while OEM's frame carries a valid request and longitudinal is allowed;
    # otherwise OEM's frame passes through unchanged. So openpilot's accel replaces the OEM ACC's accel and nothing
    # else changes on the bus: the same sender, counters and status the VCU sees under stock ACC.
    #
    # Commanding needs the OEM ACC engaged and commanding (OEM AccelVld = 1, which also means its 0x117 says Active),
    # openpilot longitudinal active, and the driver off the gas (then OEM's own frame, its override handling, goes
    # through untouched).
    if self.CP.openpilotLongitudinalControl:
      # The VCU withdrew VCU_ACCRdy mid-engagement (carstate.vcu_acc_rdy_lost): it is not taking requests, so stop
      # commanding. accFaulted disengages openpilot as well.
      long_active = CC.longActive and secoc_ok and not CS.vcu_acc_rdy_lost
      gas_override = CS.out.gasPressed
      accel_active = long_active and not gas_override and int(getattr(CS, "oem_121_vld", 0)) == 1
      # Jerk-limited request (apply_fisker_accel_limits): 0.0 at once when not commanding; on the first commanding
      # frame the speed-derived measured accel, so there is no step at engage or after a gas override; then at most
      # ACCEL_JERK_UP / ACCEL_JERK_DOWN per frame toward openpilot's accel.
      accel = apply_fisker_accel_limits(actuators.accel, self.accel_last, CS.a_ego_wheel,
                                        accel_active, self.was_accel_active_prev)
      self.accel_last = accel
      self.was_accel_active_prev = accel_active
      if long_active:
        self.long_ever_active = True
      if accel_active:
        long_reset = oem_reset_value(reset, int(CS.oem_121_reset_flag))
        accel_base = self.fcan.create_accel_command(accel, True, 0, rsv=int(getattr(CS, "oem_121_rsv", 0)))
        cands = lat_candidates(int(CS.oem_121_secoc_full_ctr), int(CS.oem_121_alive), long_reset)
        frames = [(self.fcan.steering_control_with_alive(accel_base, alive_k)[1], trip, reset_k, secoc_k)   # generic: alive + checksum
                  for secoc_k, alive_k, reset_k in cands]
        for stamped in stamp_secoc_many(self.secoc_key, ACCEL_CAN_ID, frames):
          can_sends.append((LONG_CANDIDATE_ADDR, stamped, LAT_CANDIDATE_BUS))
      self.was_tx_long_prev = accel_active

    # ---- ICC SVS / BSD / APA settings (0x35B, ICC on bus 0 -> ADAS module on bus 2) ----
    # Optional: only relayed when ICC_0x35B_OVERRIDES has entries. Panda blocks the ICC's own
    # 0x35B only while our copies are arriving, so with no overrides it's forwarded untouched.
    if ICC_0x35B_OVERRIDES:
      for icc_values in CS.icc_0x35b_frames:
        can_sends.append(self.fcan.create_icc_0x35b(icc_values))

    # ---- ICC 0x52A spoof (bus 2) ----
    # Replace the OEM ICC settings frame flowing bus 0 -> bus 2 with a mutated copy
    # that forces byte 4 to 0x5B (ICCACCSwt=1 + ICCACCFuncTyp=2 + 3 more — see
    # create_icc_spoof_0x52a). Everything else (AliveCounter, reserved bits, other
    # settings) passes through byte-identical, and the CheckSum is patched via XOR
    # delta (no need to know 0x52A's DataID).
    #
    # IMPORTANT: this must be sent UNCONDITIONALLY, not gated on
    # cruiseState.enabled. ADAS stays at ACC_Off until it sees ICC_ACCSwt=1 (and the
    # other byte-4 fields) in a valid 0x52A — i.e. the spoof is what enables ACC to
    # transition out of Off into Standby/Active. Gating on cruiseState.enabled was
    # chicken-and-egg: cruise can't become enabled without the spoof going out
    # first, and the spoof was never sent because cruise wasn't enabled.
    #
    # Cadence: one spoof per OEM tick (detected by AliveCounter change) so we never
    # emit duplicates that would trip ADAS's E2E monotonic-counter check. ICC runs
    # at ~5 Hz; openpilot runs at 100 Hz, so most carcontroller ticks won't fire.
    #
    # Panda's fisker_fwd_hook blocks the OEM's bus-0 -> bus-2 forwarding of 0x52A
    # whenever openpilot has recently emitted its own copy (time-gated, not
    # controls_allowed-gated — same reason as above).
    if CS.icc_52a_seen and CS.icc_52a_alive != self.last_spoofed_icc_alive:
      # Honor the sunnypilot UI toggles:
      #   FiskerACCAutoSpeed → ACC_AUTO_SPEED_OFF flag → clears byte-4 bit 35
      #   FiskerACCTerrain   → ACC_TERRAIN_ON flag    → sets byte-6 bit 0
      # Flags are stored in whichever polarity makes "no flag" = baseline behavior.
      # ICC_0x52A_OVERRIDES (edited live from the web HUD) goes on top and wins.
      acc_auto_speed = not bool(self.CP_SP.flags & FiskerFlagsSP.ACC_AUTO_SPEED_OFF.value)
      terrain_setting = bool(self.CP_SP.flags & FiskerFlagsSP.ACC_TERRAIN_ON.value)
      can_sends.append(self.fcan.create_icc_spoof_0x52a(CS.icc_52a_values,
                                                        acc_auto_speed=acc_auto_speed,
                                                        terrain_setting=terrain_setting))
      self.last_spoofed_icc_alive = CS.icc_52a_alive

    # ---- HUD ----
    # Forwarding intercept: the OEM ADAS module stays alive on bus 2 and the panda
    # forwards its status/HUD frames (ACC HUD 0x31C, warning HUD 0x317) to the cluster,
    # so openpilot must NOT also transmit them — that would collide with the OEM's. It
    # only injects the steering command; the OEM keeps driving the cluster/HUD.

    new_actuators = actuators.as_builder()
    new_actuators.steeringAngleDeg = self.apply_angle_last
    if self.CP.openpilotLongitudinalControl:
      new_actuators.accel = self.accel_last

    self.frame += 1
    return new_actuators, can_sends
