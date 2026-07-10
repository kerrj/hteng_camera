"""A small software auto-exposure controller for HTENG cameras.

Why software AE instead of the SDK's ``CameraSetAeState``: the SDK's AE varies
*exposure time* freely, which balloons the sensor's frame interval in dim scenes
— fine for a still preview, fatal for ``record_ego`` where a stable, matched
frame interval is what keeps the timestamp-paired stereo sync tight. This
controller keeps exposure pinned short and bounded and reaches the brightness
target *gain-first*, so the frame rate stays put.

The control law (see :meth:`AutoExposure.update`):

  * Meter a robust brightness off the centre ~70% of the frame — appropriate for
    a 180° VR fisheye, whose dark vignetted corners would otherwise drag the
    meter and blow out the centre. Centre-weighted mean, normalised to 0..1.
  * Drive it as a **damped proportional controller**: correct only a fraction
    (``responsiveness``) of the brightness error in log space each update,
    instead of trying to fully correct in one step. A full ("deadbeat") step
    oscillates here, because the exposure/gain we write doesn't reach the metered
    frame for a frame or two (GET_NEWEST can return an in-flight frame), so a
    full corrector stacks the same correction several times before seeing the
    effect and then overshoots. Loop gain < 1 is what makes the loop robust to
    that latency.
  * Convert the damped correction into a desired total light = exposure × gain,
    then allocate it **gain-first**: keep exposure at its floor and raise gain;
    only once gain saturates does exposure creep up toward its cap. This is
    "prioritise low exposure, mostly adjust gain".
  * A deadband (don't act inside ±x% of target) plus a per-step clamp on the
    *total* light change keep the loop calm. Note we deliberately do NOT
    rate-limit gain on its own: when a discrete exposure step changes (e.g.
    8.33→16.67 ms), gain must jump to keep total light constant, and a gain rate
    limit would instead flash the image ~2× and hunt — the classic step-switch
    oscillation. Damping the total handles smoothness without that failure mode.

The controller is pure policy — it never touches the camera. The caller meters a
frame and applies the returned exposure/gain. It accepts either a raw Bayer
plane (H, W) or a linear RGB frame (H, W, 3); both are treated as 0..``max_val``.
"""

from __future__ import annotations

import numpy as np


class AutoExposure:
    """Gain-first software auto-exposure policy.

    Parameters
    ----------
    gain_min, gain_max : float
        Analog-gain multiplier bounds (from ``cam.gain_range()``).
    exp_min, exp_max : float
        Exposure bounds in ms. ``exp_min`` is the pinned floor the loop prefers;
        exposure only rises toward ``exp_max`` once gain saturates. Keep
        ``exp_max`` below ``1000/fps`` if a stable frame rate matters.
    target : float
        Desired centre brightness, 0..1 (fraction of full scale). ~0.3–0.4 is a
        well-exposed mid-grey.
    center_frac : float
        Fraction of width/height to meter (centred). 0.7 = central 70%.
    responsiveness : float
        Loop gain in (0, 1]: the fraction of the (log) brightness error corrected
        each update. 1.0 = full ("deadbeat") correction in one step — oscillates
        here because the metered frame lags the setting by a frame or two. ~0.2–0.3
        is a good damped value: quick but stable regardless of loop rate. THIS is
        the smoothing knob — there is no EMA in the feedback path (lag there
        destabilises, the opposite of what you'd want).
    deadband : float
        Fractional dead zone around the target (0.08 = ±8%); no change inside it.
    max_step : float
        Max fractional change in *total light* (exposure × gain) per update
        (0.5 = ±50%), a safety clamp on slew. Applied to the total, not to gain
        alone — clamping gain alone would fight the exposure step-switch and flash.
    max_val : float
        Full-scale pixel value (4095 for this 12-bit sensor).
    """

    def __init__(self, gain_min, gain_max, *, exp_min=5.0, exp_max=17.0,
                 target=0.35, center_frac=0.7, responsiveness=0.25, deadband=0.08,
                 max_step=0.5, max_val=4095.0, flicker_hz=60.0,
                 step_hysteresis=0.1):
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)
        self.exp_min = float(exp_min)
        self.exp_max = float(exp_max)
        self.target = float(target)
        self.center_frac = float(center_frac)
        self.responsiveness = float(responsiveness)
        self.deadband = float(deadband)
        self.max_step = float(max_step)
        self.max_val = float(max_val)
        #: Gain headroom (fraction) a shorter exposure must have to spare before
        #: the loop steps DOWN to it. Stepping up to a longer exposure is
        #: immediate (when gain saturates); stepping back down waits for this much
        #: margin so the choice is biased toward low exposure without flip-flopping
        #: at the boundary. 0.1 = drop to 8.33 ms once it fits with ≥10% gain spare.
        self.step_hysteresis = float(step_hysteresis)
        #: Mains frequency for anti-flicker, Hz (60 in US / 50 in EU), or 0/None
        #: to disable. When set, exposure is restricted to integer multiples of
        #: the light's intensity period (1000/(2·flicker_hz) ms — light flickers
        #: at twice mains) that fall in [exp_min, exp_max], so each frame
        #: integrates whole flicker cycles and banding/pulsing cancels optically.
        #: All fine brightness control then falls to gain.
        self.flicker_hz = float(flicker_hz) if flicker_hz else 0.0
        # cached centre-weight kernel, rebuilt when the metered shape changes
        self._wkey = None
        self._wy = None
        self._wx = None
        # Deferred light-increasing write (see apply): (value, kind, frames_left).
        self._pending = None

    # -- metering ----------------------------------------------------------

    def _weights(self, h, w):
        """Separable triangular centre weighting for an (h, w) region, cached."""
        if self._wkey != (h, w):
            # Triangular falloff to the edges: 1.0 at centre, 0.3 at the border.
            wy = 1.0 - 0.7 * np.abs(np.linspace(-1.0, 1.0, h, dtype=np.float32))
            wx = 1.0 - 0.7 * np.abs(np.linspace(-1.0, 1.0, w, dtype=np.float32))
            self._wy, self._wx, self._wkey = wy, wx, (h, w)
        return self._wy, self._wx

    def meter(self, img, stride=4):
        """Centre-weighted mean brightness of ``img``, normalised to 0..1.

        ``img`` may be a Bayer plane (H, W) or linear RGB (H, W, 3). ``stride``
        subsamples for speed (metering doesn't need every pixel).
        """
        h, w = img.shape[:2]
        cf = self.center_frac
        y0, y1 = int(h * (1 - cf) / 2), int(h * (1 + cf) / 2)
        x0, x1 = int(w * (1 - cf) / 2), int(w * (1 + cf) / 2)
        region = img[y0:y1:stride, x0:x1:stride]
        if region.ndim == 3:
            # Rec.709 luma; cheap and good enough for metering.
            region = (0.2126 * region[..., 0] + 0.7152 * region[..., 1]
                      + 0.0722 * region[..., 2])
        region = region.astype(np.float32)
        wy, wx = self._weights(region.shape[0], region.shape[1])
        # Separable weighted mean: (wy · region · wx) / (sum wy)(sum wx).
        num = wy @ region @ wx
        den = wy.sum() * wx.sum()
        return float(num / den) / self.max_val

    # -- control -----------------------------------------------------------

    def reset(self):
        """No persistent loop state to clear (the controller is memoryless); kept
        for API symmetry / future use."""
        return

    def _exposure_steps(self):
        """The exposures the loop is allowed to use, ascending.

        With anti-flicker on, these are the integer multiples of the light's
        intensity half-period (1000/(2·flicker_hz) ms) lying in
        [exp_min, exp_max] — e.g. 60 Hz → 8.33, 16.67 ms. Each integrates whole
        flicker cycles, so mains banding/pulsing cancels. Falls back to a single
        free-floating exposure (the floor) when no multiple fits or anti-flicker
        is off.
        """
        if self.flicker_hz > 0:
            period = 1000.0 / (2.0 * self.flicker_hz)   # light flickers at 2× mains
            n_lo = int(np.ceil(self.exp_min / period - 1e-6))
            n_hi = int(np.floor(self.exp_max / period + 1e-6))
            steps = [round(n * period, 4) for n in range(max(1, n_lo), n_hi + 1)]
            if steps:
                return steps
        return [self.exp_min]   # anti-flicker off, or no multiple fits the range

    def update(self, img, cur_exp, cur_gain):
        """Meter ``img`` and return ``(new_exp_ms, new_gain_x, info)``.

        ``cur_exp``/``cur_gain`` MUST be the exact values currently applied to the
        camera — feed back what you last applied, never a rounded/quantised copy,
        or the loop chases its own rounding error.

        ``info`` carries ``measured``/``error``/``changed`` for display.
        ``new_exp``/``new_gain`` always reflect the chosen operating point; inside
        the deadband they equal the current values and ``changed`` is False.
        """
        measured = self.meter(img)
        info = {"measured": measured, "error": 0.0, "changed": False,
                "exp": cur_exp, "gain": cur_gain}

        # Work in log space so exposure and gain (both multiplicative on
        # brightness) compose linearly and the loop behaves the same bright or dark.
        # err > 0 means too dark (need more light). Guard a black frame.
        m = max(measured, 1e-4)
        err = np.log(self.target / m)
        info["error"] = float(err)

        # Deadband around the target (in log units ≈ fractional brightness error).
        if abs(err) <= self.deadband:
            return cur_exp, cur_gain, info

        # Damped proportional step: correct only `responsiveness` of the error.
        # Loop gain < 1 is what keeps the loop stable despite the 1–2 frame lag
        # between writing a setting and metering its effect (the cause of the
        # deadbeat oscillation). Clamp the per-step change in TOTAL light so a big
        # scene change slews in rather than slamming.
        step_log = self.responsiveness * err
        cap = np.log1p(self.max_step)                # ±max_step on total light
        step_log = float(np.clip(step_log, -cap, cap))
        total = cur_exp * cur_gain * np.exp(step_log)

        # Gain-first allocation over the allowed (flicker-free) exposure steps.
        # Prefer the SHORTEST exposure that can hold the needed light within the
        # gain range (least motion blur) — `want` is that ideal pick.
        steps = self._exposure_steps()
        want = steps[-1]                             # fallback: longest available
        for s in steps:
            if total / s <= self.gain_max:
                want = s
                break

        # Asymmetric hysteresis, biased toward LOW exposure:
        #   * step UP to a longer exposure the instant the shorter one saturates
        #     gain (want > cur) — we genuinely need the light;
        #   * step DOWN to a shorter exposure as soon as it fits with a little gain
        #     headroom to spare (margin), so we return to 8.33 ms promptly instead
        #     of lingering at 16.67;
        #   * the margin is the only reluctance — it stops brightness hovering at
        #     the boundary from flip-flopping every frame.
        # Match cur_exp to a step tolerantly (caller may pass a rounded value).
        cur_step = min(steps, key=lambda s: abs(s - cur_exp))
        on_grid = abs(cur_step - cur_exp) <= 0.05
        if not on_grid or want >= cur_step:
            exp = want                               # going up (needed) or staying
        elif total <= want * self.gain_max * (1.0 - self.step_hysteresis):
            exp = want                               # shorter exposure fits comfortably
        else:
            exp = cur_step                           # too close to the edge — hold
        gain = min(self.gain_max, max(self.gain_min, total / exp))

        info.update(changed=True, exp=exp, gain=gain)
        return exp, gain, info

    #: Frames to hold a light-increasing write after its light-reducing partner,
    #: so the reducing one has surely taken effect first. The two settings have
    #: DIFFERENT take-effect latencies (measured on this hardware: gain ~1 frame,
    #: exposure ~2 frames), so merely *ordering* the USB writes isn't enough — the
    #: faster-latching one can still land first and flash. Holding 2 frames covers
    #: the slower (exposure) latency.
    DEFER_FRAMES = 2

    def apply(self, cams, new_exp, new_gain, cur_exp, cur_gain):
        """Apply ``(new_exp, new_gain)`` to one or more cameras flash-free.

        When an 8.33↔16.67 ms exposure step flips and gain compensates the other
        way, total light should stay constant — but exposure and gain are two
        separate USB writes with *different take-effect latencies* (measured: gain
        ~1 frame, exposure ~2 frames). So even issuing them in the right order, the
        faster-latching setting can land a frame before its partner and flash ~2×
        bright. We make any transient go DARK instead: apply the light-*reducing*
        write immediately and DEFER the light-*increasing* one by
        :data:`DEFER_FRAMES` calls, by which point the reducing one has taken
        effect. While a write is deferred we also FREEZE new commands, so the
        controller can't react to the (dark) transitional frames and cancel the
        pending raise — the step-switch behaves as one atomic move over a couple
        of frames. Worst case is a couple of slightly-dim frames, never a flash.

        Call once per captured frame (it counts the deferral down). ``cams`` is a
        single HTCamera or an iterable (the stereo pair, driven from one meter so
        L/R stay matched). ``cur_exp``/``cur_gain`` are the values currently in
        effect. Returns the ``(exp, gain)`` now actually commanded on the camera —
        feed THAT back as next frame's ``cur_*`` (during a deferral the increasing
        setting still reads as its old value, which is the truth).
        """
        try:
            cam_list = list(cams)
        except TypeError:
            cam_list = [cams]

        def write(kind, value):
            for c in cam_list:
                if kind == "exp":
                    c.set_exposure_ms(value)
                else:
                    c.set_analog_gain(value)

        # While a deferred write is outstanding, ignore new commands (freeze) and
        # just count it down; apply it when due. Returns the values truly in effect.
        if self._pending is not None:
            value, kind, left = self._pending
            left -= 1
            if left <= 0:
                write(kind, value)
                self._pending = None
                # The deferred (increasing) write now lands; its partner cut is
                # already in effect, so both targets are now reached.
                if kind == "exp":
                    return value, cur_gain
                return cur_exp, value
            self._pending = (value, kind, left)
            return cur_exp, cur_gain

        exp_changed = abs(new_exp - cur_exp) > 1e-4
        gain_changed = abs(new_gain - cur_gain) > 1e-3
        if not (exp_changed or gain_changed):
            return cur_exp, cur_gain

        # Single setting changed: just write it (no two-write race to manage).
        if exp_changed != gain_changed:
            kind, value = ("exp", new_exp) if exp_changed else ("gain", new_gain)
            write(kind, value)
            return new_exp, new_gain

        # Both change (the exposure step-switch). Apply whichever write *reduces*
        # light now; DEFER the one that *increases* light. Decided per-setting, not
        # on net total: even when net light falls, the gain-raise latches before
        # the exposure-cut and would flash bright. The deferred raise lands once
        # its partner cut has taken effect, so the transient is always a dim frame.
        if new_exp < cur_exp:
            # Exposure drops (16.67→8.33, −light); gain rises (+light): cut now,
            # defer the gain raise. In effect this frame: new_exp, old gain.
            write("exp", new_exp)
            self._pending = (new_gain, "gain", self.DEFER_FRAMES)
            return new_exp, cur_gain
        # Exposure rises (8.33→16.67, +light); gain drops (−light): cut gain now,
        # defer the exposure raise. In effect this frame: old exp, new_gain.
        write("gain", new_gain)
        self._pending = (new_exp, "exp", self.DEFER_FRAMES)
        return cur_exp, new_gain
