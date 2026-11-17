from dataclasses import dataclass, field
from enum import IntFlag

from opendbc.car import Bus, CarSpecs, DbcDict, PlatformConfig, Platforms
from opendbc.car.lateral import AngleSteeringLimits
from opendbc.car.structs import CarParams
from opendbc.car.docs_definitions import CarDocs, CarHarness, CarParts
from opendbc.car.fw_query_definitions import FwQueryConfig

Ecu = CarParams.Ecu


@dataclass
class FiskerCarDocs(CarDocs):
  package: str = "All"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.custom]))


@dataclass
class FiskerPlatformConfig(PlatformConfig):
  # Everything openpilot needs is on ADASBUS — the gateway mirrors the body/HMI
  # signals (gear, pedal, doors, seatbelt, blinkers, MFSS buttons) onto it — so
  # the car state comes from a single DBC (there is no separate body DBC). The
  # radar tracks come from the MRR's own link on panda bus 1.
  dbc_dict: DbcDict = field(default_factory=lambda: {
    Bus.pt: 'fisker_ocean_adas',
    # Mid-range radar private link, panda bus 1. Reverse-engineered; see the DBC's comments.
    Bus.radar: 'fisker_ocean_mrr',
  })


class CAR(Platforms):
  FISKER_OCEAN = FiskerPlatformConfig(
    [FiskerCarDocs("Fisker Ocean 2023-24")],
    CarSpecs(
      mass=2410.,                # Ocean Extreme curb + typical equipment
      wheelbase=2.921,           # 2921 mm per published Ocean data sheet (was 2.99,
                                 # made the yaw model expect slower turn-in and
                                 # caused MPC to over-command angle on curve entry)
      steerRatio=15.0,           # first-cut; refine from step-response logs
      # centerToFrontRatio kept at default 0.5 — battery-in-floor EV is roughly 50/50.
      tireStiffnessFactor=0.85,  # crossover-typical; default 1.0 assumed sportier tires,
                                 # so MPC expected more grip than the car has → overshoot.
    ),
  )


class CANBUS:
  # Camera-splice harness: panda sits between the vehicle-side gateway (bus 0) and the
  # OEM ADAS module (bus 2). Bus 1 is the mid-range radar's private link, when the harness
  # taps it (see radar_interface.py). Verified from a live fingerprint capture:
  # bus 0 has 92 addresses (gateway-mirrored body/HMI + ESP/EPS/YRS/BCM/VCU/MFS/ICC),
  # bus 2 has ~65 addresses (all ADAS-authored: 0x117/0x118/0x121/0x1C0/0x1D0/0x313/
  # 0x314/0x315/0x316/0x317/0x31A/0x31C + ADAS radar object list 0x2C7..0x2E9).
  pt = 0           # ADASBUS, vehicle side (CAN-FD)
  cam = 2          # ADASBUS, OEM ADAS module side (CAN-FD)
  radar = 1        # MRR private link (CAN-FD), receive only: safety never forwards or sends on it


# VCU_GearSig enum (from DBC VAL_): 1=P, 2=N, 3=R, 4=D, 6=E, 7=S
GEAR_MAP = {
  0: CarParams.Ecu.unknown,  # placeholder; mapped to GearShifter in carstate.py
}


# MFSS button mapping
# Both MFS_RiBtnNorth and MFS_RiBtnEast engage the car's ACC on the vehicle side (VCU/ICC
# treat them identically for cc_state transitions). On this port they differ in whether
# openpilot MADS auto-engages lateral alongside the cruise→Active transition:
#   MFS_RiBtnNorth  → ACC only (no MADS)   — default/safer, see carstate's mads_arm tracker
#   MFS_RiBtnEast   → ACC + MADS (full UEM behavior, matches older ports)
# The two still emit the same ButtonType.mainCruise event; the per-button gating lives in
# fisker.carstate (which writes the FiskerMadsArmed param) and sunnypilot/mads/mads.py
# (which reads it in block_unified_engagement_mode).
BUTTON_MAP = {
  "MFS_RiBtnNorth":  "mainCruise",   # ACC main — engages ACC only (MADS stays off)
  "MFS_RiBtnEast":   "mainCruise",   # ACC main — engages ACC + arms MADS for UEM
  "MFS_LeRollPress": "setCruise",    # engage cruise
  "MFS_LeRollUp":    "accelCruise",  # +speed
  "MFS_LeRollDwn":   "decelCruise",  # -speed
  "MFS_RiBtnSouth":  "lkas",         # MADS (sunnypilot) toggle — LONG PRESS only (see
                                     #   carstate._is_pressed), short press is used by
                                     #   the car for ACC follow-distance adjustment.
  # cancel / resume / gapAdjustCruise: TBD from bench testing
}

# DBC VAL_ for these signals is 0=No_Pressed 1=Pressed 2=Long_Press 3=Reserved — a single
# signal carries both tap and hold as distinct enum values (not two signals). Buttons listed
# here only fire their ButtonEvent on the Long_Press (2) state; a short Pressed (1) is ignored.
BUTTON_LONG_PRESS_ONLY = {"MFS_RiBtnSouth"}


class CarControllerParams:
  # Steering angle command — ADAS_LatCtrl_SteerAnReq in 0x1D0
  # Signal range ±780°, 0.0625°/LSB. Cycle time 10 ms (100 Hz).
  # Fisker's carcontroller uses apply_std_steer_angle_limits (not the VM variant),
  # so ANGLE_LIMITS only carries STEER_ANGLE_MAX + up/down rate lookups.
  ANGLE_LIMITS: AngleSteeringLimits = AngleSteeringLimits(
    600,                                  # STEER_ANGLE_MAX (deg) — DBC permits ±780, conservative 600
    ([0., 5., 25.], [2.5, 1.5, 0.2]),     # ANGLE_RATE_LIMIT_UP (deg/frame by m/s)
    ([0., 5., 25.], [5.0, 2.0, 0.3]),     # ANGLE_RATE_LIMIT_DOWN
  )

  STEER_STEP = 1                          # 0x1D0 sent every frame (10 ms ≈ 100 Hz) — verify cycle from logs
  # Capped at 1.0 (was 2.0): the stock ACC never asked for more than ~0.9 m/s2 in recorded drives, and in
  # 000000c6--51d13b87bd--1 the car faulted itself (VCU_ACCRdy dropped, ACC Failure, ESP/AEB faults) after
  # accelerating at ~1.5 m/s2 for ~0.6 s, 0.4-0.6 above our request (which had peaked at +1.29), most
  # likely tripping an actual-vs-requested acceleration plausibility check. Panda enforces the same cap
  # (FISKER_ACCEL_MAX in safety/modes/fisker.h).
  ACCEL_MAX = 1.0                         # m/s²
  ACCEL_MIN = -3.5                        # m/s²

  # Jerk (slew) limits on the 0x121 accel request, per direction, in m/s^3 (the carcontroller applies
  # them per 10 ms frame). From the stock ADAS's own 0x121 requests while stock ACC was Active (OEM
  # bus-2 frames in the 6 recorded routes under routes/, 4.9 min Active incl. the braking in 000000dd and 000000e8):
  # the stock ADAS updates its request in steps every ~30 ms, so it was measured over 100 ms windows,
  # skipping 0.5 s after activation and 0.2 s before deactivation. Rising: p99 1.50, p99.9 14.5.
  # Falling: p99 2.80, p99.9 15.0. Both limits are set below the stock p99. Rising is also kept at
  # 1.0 or lower to stay well away from the actual-above-request fault in 000000c6 (see ACCEL_MAX).
  ACCEL_JERK_UP = 1.0                     # m/s^3, accel request increasing (incl. releasing the brake)
  ACCEL_JERK_DOWN = 2.5                   # m/s^3, accel request decreasing (incl. braking harder)

  # 0x121 ADAS_LgtCtrl_AccelReq scale: 0.0004882 m/s²/LSB, offset -16
  ACCEL_SCALE = 0.0004882
  ACCEL_OFFSET = -16.0


class FiskerSafetyFlags(IntFlag):
  LONG_CONTROL = 1
  SECOC = 2


class FiskerFlags(IntFlag):
  LONG_CONTROL = 1


# SecOC-protected message IDs on ADASBUS (from DBC scan).
# DataId in the AUTOSAR Short SecOC profile equals the CAN message ID for these.
SECOC_TX_MSGS = {
  0x1D0: "ADAS_0x1D0",   # steering angle command
  0x121: "ADAS_0x121",   # accel command
}
SECOC_SYNC_MSG = (0x20, "GW_Syn_All")    # 50 ms, full 40-bit freshness on wire


# ICC_0x52A: the ICC's feature settings, re-sent to the ADAS module (bus 2) on every ICC frame
# (fiskercan.create_icc_spoof_0x52a) with these signals replaced. Everything else, including the
# E2E AliveCounter, passes through from the ICC's frame and the CheckSum is patched. The defaults
# below are what enable ACC (byte 4 = 0x5B with auto speed on). ICCACCAutoSpdSts and
# ICCACCTerrainSetting default to the on-device toggles (FiskerACCAutoSpeed, FiskerACCTerrain); an
# entry here wins over them. Edited live from the web HUD (Car → Assist, or CAN settings). Raw
# values; names from the DBC.
ICC_0x52A_OVERRIDES: dict[str, float] = {
  "ICCACCSwt": 1,                 # 0=Invalid 1=On 2=Off -- ACC master switch; ADAS stays ACC_Off without On
  "ICCACCFuncTyp": 2,             # 0=Default 1=Basic 2=Advanced 3=ISA
  "ICCACCSpdStepSize": 1,         # 0=Step_1_unit 1=Step_5_unit
  "ICCLaneTrajectorySetting": 1,  # 0=Off 1=On
  # "ICCACCAutoSpdSts": 1,          # 0=Off 1=On -- set speed follows the speed limit
  # "ICCACCTerrainSetting": 0,      # 0=Not_enabled 1=Enabled
  # "ICCACCSpdLimOffsTyp": 1,       # 0=Exceeding_limit_not_allowed 1=Percentage 2=Speed_value
  # "ICCACCSpdLimOffs": 5,          # 0..31.5, step 0.5: % or speed over the limit, per the type
}

# Driver acknowledgments (time-gap prompt, ISA warning) and the undefined bits -- always passed
# through from the ICC's frame.
ICC_0x52A_PASSTHROUGH = frozenset({
  "ICCACCTiGapCfm",
  "ICCISAWarnStopReq",
  "ICC_0x52A_Rsv49",
  "ICC_0x52A_Rsv55",
})
assert not (ICC_0x52A_OVERRIDES.keys() & ICC_0x52A_PASSTHROUGH), "ICC_0x52A pass-through signals can't be overridden"


# ICC_0x35B: surround-view (SVS) requests + BSD/DOW/DCAA/APA/park-assist/AHBA settings. No E2E
# (no checksum/counter). Optional: while this dict is empty openpilot leaves 0x35B alone and
# panda forwards the ICC's frame as-is. Once any override is set, openpilot re-sends every ICC
# frame to the ADAS module (bus 2) with these signals replaced; everything else, including the
# undocumented bits, passes through unchanged. Raw values; names from the DBC VAL_ tables.
ICC_0x35B_OVERRIDES: dict[str, int] = {
  "ICC_EnbLnChgAsst": 1,          # 0=Auto_steer_not_enabled 1=Auto_steer_enabled
  # "ICC_BSD_Sensitivity": 0,       # 0=Normal 1=Early 2=Late_or_reduced
  # "ICC_DOW_Setting": 2,           # 0=OFF 1=ON_with_Visual 2=ON_with_visual_and_audio
  # "ICC_DCAASetting": 1,           # 0=OFF 1=ON
  "ICC_AHBA_Setting": 1,          # 0=Off 1=On
  # "ICC_WarnTypeSetting": 2,       # 0=OFF 1=ON_with_Visual 2=ON_with_visual_and_audio
  #                                 #   3=ON_with_visual_and_audio_and_Steering_Wheel_Vibration
  # "ICC_LKA_SettingWrnTyp": 1,     # 0=Audio_Visual_and_Haptic 1=Visual_and_Audio
  "ICC_SteerWhlVibrSet": 1,       # 0=Off 1=On
  "ICC_APA_Setting": 2,           # 0=Coded_off_(not_equipped) 1=Off_(by_user) 2=Enabled
  "ICC_APAParkInDirSetting": 1,   # 0=Nose-in 1=Back-in
  "ICC_APAParkOutDirSetting": 1,  # 0=Right 1=Left
  "ICC_ParkAsstChmAlrt": 1,       # 0=Off 1=On
  "ICC_WSPPA_Setting": 1,         # 0=No_curb_protection 1=Warning_only 2=Braking_intervention
  "ICC_RAP_Setting": 1,           # 0=Remote_Automated_Parking_Disabled 1=..._Enabled
  "ICC_TP_Setting": 1,            # 0=Trained_Parking_Disabled 1=Trained_Parking_Enabled
  # Also in this frame but momentary user actions, not settings — overriding one holds it
  # asserted on every frame: ICC_APAParkSelect (slot), ICC_APAActivation (1=On starts APA),
  # ICC_APAParkOutAct (1=Request starts park-out). They are listed in ICC_0x35B_MOMENTARY and
  # must never be a default here.
}

# Momentary user actions in ICC_0x35B (see above). Not allowed in the code defaults; the ICC's
# own value passes through unless someone sets one live from the web HUD.
ICC_0x35B_MOMENTARY = frozenset({
  "ICC_APAParkSelect",
  "ICC_APAActivation",
  "ICC_APAParkOutAct",
})
assert not (ICC_0x35B_OVERRIDES.keys() & ICC_0x35B_MOMENTARY), "ICC_0x35B momentary actions can't be default overrides"

# Live user interactions with the camera view — always passed through from the ICC's frame.
ICC_0x35B_PASSTHROUGH = frozenset({
  "ICC_ViewReq",
  "ICC_ShowSts",
  "ICC_SVS_AutoVwBttnReq",
  "ICC_SVS_CamVwCnclReq",
  "ICC_AutomaticViewReq",
  "ICC_360FreeFloatingReq",
  "ICC_GraphicOverlayReq",
  "ICC_OperationType",
  "ICC_ZOOM_Start",
  "ICC_ZOOM_End",
  "ICC_ZOOM_ScalingFactor",
  # blind spot: the head unit's own setting. Relayed as 3 (visual + audio + wheel vibration) the ADAS
  # reported BSD Off for a whole drive (route 00000028, 2 Nov 2026), standby before the relay began.
  "ICC_BSDSetting",
  "ICC_BSD_Sensitivity",
})
assert not (ICC_0x35B_OVERRIDES.keys() & ICC_0x35B_PASSTHROUGH), "ICC_0x35B pass-through signals can't be overridden"


# No FW-version fingerprinting for Fisker: FW_VERSIONS is intentionally empty
# (single-model port, CAN fingerprint is authoritative), so declare zero requests
# to keep the brand out of fw_versions.get_brand_ecu_matches — otherwise upstream's
# `len(brand_matches[b])` divide-by-zero crashes card at startup. The fw_version_regex
# is not used when requests is empty; supply a matches-nothing pattern for the record.
FW_QUERY_CONFIG = FwQueryConfig(requests=[], fw_version_regex=rb'.*')


DBC = CAR.create_dbc_map()

STEER_THRESHOLD = 0.5  # Nm — driver torque to count as "hands on" (steeringPressed, ALC/UI)

# Driver torque at which we proactively yield lateral (drop the request so the driver can
# steer). Tunable. On-vehicle 2026-10-26 the EPS's own intervention detector only trips at
# ~6.5 Nm, which is a lot of effort and lets the EPS fight up to that point before releasing
# all at once (the wheel "jump"). Driver torque (EPS_DrvrSteerTq) reads ~0.0-0.3 Nm hands-off,
# so 3.0 Nm sits well clear of the floor while being < half the EPS trip: easier override, and
# yielding early means little stored-up fight to release. Raise it if it ever yields on its
# own (e.g. a sharp curve); lower it for an even lighter override.
OVERRIDE_STEER_TORQUE = 3.0  # Nm
# ...held for OVERRIDE_TRIP_FRAMES in a row (~100 Hz). A bump jolts the column past 3 Nm for
# 10-80 ms (2026-11-02: at least 6 times, each dropping steering for 0.7-2.3 s and two leading to
# an LKAS fault on the way back in); a real grab held it 250 ms or more. The EPS's own ~6.5 Nm
# intervention bit still releases at once.
OVERRIDE_TRIP_FRAMES = 10      # ~0.1 s
# Hysteresis for the torque yield: once latched, only resume after driver torque drops below
# OVERRIDE_RELEASE_TORQUE and stays there OVERRIDE_RELEASE_FRAMES (~100 Hz). Prevents the
# grab/release chatter when the driver holds steady near the trip point.
OVERRIDE_RELEASE_TORQUE = 0.5  # Nm — resume gate (above the ~0.3 Nm hands-off floor; tuned on-car)
OVERRIDE_RELEASE_FRAMES = 25   # ~0.25 s

# Pause lateral actuation below this speed (keep the EPS fed with Req=0) so the wheel
# doesn't creep/center while stopped. 0.447 m/s = 1 mph.
STANDSTILL_PAUSE_SPEED = 0.447  # m/s

# Settle wait on resume: don't re-grab the wheel until it has stopped self-centering. Hold
# while it moves faster than SETTLE_ANGLE_STEP deg per ~100 Hz tick (~0.2 = ~20 deg/s),
# bounded by SETTLE_MAX_TICKS (~1.5 s) so it can't wait forever.
SETTLE_ANGLE_STEP = 0.2   # deg/tick
SETTLE_MAX_TICKS = 150    # ~1.5 s

# The most wheel angle openpilot itself asks for (deg). The EPS aborts near full lock (~470 deg): on 2026-10-27 and on
# route 00000103 (2026-11-16, a 5 mph left turn: the model asked for 460 deg with the wheel at 445, AbortFb=9 and a
# permanent LKAS fault, and the car's own cruise then pulled away at +2.5 m/s^2). The turn assist's tightest arc needs
# ~280 deg. A driver who wants more turns the wheel past this and the angle-divergence override yields to them. Only the
# command is capped: while not steering, the angle follows the measured wheel wherever it is (the panda checks that).
LAT_ANGLE_CMD_MAX = 400.0          # deg

# Command trim. The wheel settles about 0.85 deg left of the 0x1D0 angle we send, at every speed and command size
# (wheel minus command, median over |command| < 2 deg: +0.95 f5, +0.75 fe, +0.93 101, +0.82 105, +0.79 106). openpilot's
# angle controller has no integral, so the car went straight with the wheel at +0.5 deg while openpilot asked for
# -0.3 deg, and sat 30 cm left of the lane centre on the freeway (route 00000106, 2026-11-17: lane centre +0.33 m right
# of the car under openpilot, +0.02 m when the driver steered). We send the command minus this trim, and keep learning
# it slowly while openpilot holds the wheel steady at small angles, within MIN..MAX.
ANGLE_TRIM_DEG = 0.85              # deg, starting value
ANGLE_TRIM_MIN = 0.3               # deg
ANGLE_TRIM_MAX = 1.5               # deg
ANGLE_TRIM_TAU_S = 20.0            # s of steady driving to move most of the way to a new value
ANGLE_TRIM_MIN_SPEED = 9.0         # m/s: below this the wheel moves too much to learn from
ANGLE_TRIM_MAX_CMD = 3.0           # deg: learn only near straight ahead
ANGLE_TRIM_STEADY_STEP = 0.005     # deg per 10 ms: the command must be this still (lag error under 0.1 deg)
ANGLE_TRIM_MAX_JUMP = 1.0          # deg: a sample this far from the current trim is not the offset (ignored)

# Angle-divergence override: when the measured wheel is turned this far (deg) from the angle
# the model is commanding, the driver is clearly steering independently (hard manual input /
# U-turn) even if column torque stayed under OVERRIDE_STEER_TORQUE because the EPS was power-
# assisting the turn. Yield rather than keep commanding into it (on-vehicle 2026-10-27 that
# caused a near-full-lock EPS abort=9 and a permanent LKAS fault). Hysteresis: latch on at
# OVERRIDE_DEG, release once the wheel is back within RELEASE_DEG of the command.
ANGLE_OVERRIDE_DEG = 30.0          # deg
ANGLE_OVERRIDE_RELEASE_DEG = 10.0  # deg
# Cooperative signaled turn (carcontroller.coop_signaled_turn): below this speed, with the blinker on toward the
# side the wheel is turned, a command wanting at least as much turn as the wheel has is not an override. Matches
# the map turn planner's release speed (turn_planner.V_ABORT).
TURN_COOP_MAX_SPEED = 9.0          # m/s (~20 mph)
TURN_COOP_UNWIND_DEG = 0.05        # deg/tick (5 deg/s): the wheel coming back faster than this is the driver unwinding
