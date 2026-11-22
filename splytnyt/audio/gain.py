"""Gain planning.

This module **plans**; it never applies anything. The plan it returns is
shown to the user, who decides whether to use the gain, use less gain, or
explicitly allow a limiter.

The product rule is simple: if a loudness target cannot be reached without
exceeding the true-peak ceiling, SplytNyt must say so and require an
explicit decision. It must never silently limit.
"""

from __future__ import annotations

from dataclasses import dataclass

from .measure import LoudnessResult


@dataclass
class GainPlan:
    """A proposed gain, with the trade-off spelled out."""

    mode: str = "shared"
    #: The gain that would actually be applied.
    gain_db: float = 0.0
    #: False when the target cannot be met under the ceiling.
    feasible: bool = True
    #: Predicted true peak after applying ``gain_db``.
    peak_after_dbtp: float = -120.0
    #: The largest gain that still respects the ceiling.
    safe_gain_db: float = 0.0
    ceiling_dbtp: float = -1.0
    target_lufs: float = -14.0
    #: True when reaching the target needs a limiter.
    required_limiting: bool = False
    #: Human-readable explanation for the UI.
    reason: str = ""

    def linear(self) -> float:
        """The gain as a linear amplitude multiplier."""
        if self.gain_db <= -120.0:
            return 0.0
        return 10.0 ** (self.gain_db / 20.0)


def _plan(
    measured: LoudnessResult,
    target_lufs: float,
    ceiling_dbtp: float,
    mode: str,
) -> GainPlan:
    if measured.is_silent():
        return GainPlan(
            mode=mode,
            gain_db=0.0,
            feasible=True,
            peak_after_dbtp=measured.true_peak_dbtp,
            safe_gain_db=0.0,
            ceiling_dbtp=ceiling_dbtp,
            target_lufs=target_lufs,
            required_limiting=False,
            reason="The audio is silent, so no gain is applied.",
        )

    requested = target_lufs - measured.integrated_lufs
    peak_after = measured.true_peak_dbtp + requested
    safe_gain = ceiling_dbtp - measured.true_peak_dbtp
    safe_gain = min(safe_gain, requested)

    if peak_after <= ceiling_dbtp:
        return GainPlan(
            mode=mode,
            gain_db=requested,
            feasible=True,
            peak_after_dbtp=peak_after,
            safe_gain_db=safe_gain,
            ceiling_dbtp=ceiling_dbtp,
            target_lufs=target_lufs,
            required_limiting=False,
            reason=(
                f"Applying {requested:+.1f} dB brings the recording to "
                f"{measured.integrated_lufs + requested:.1f} LUFS with a true peak of "
                f"{peak_after:.1f} dBTP, under the {ceiling_dbtp:.1f} dBTP ceiling."
            ),
        )

    return GainPlan(
        mode=mode,
        gain_db=requested,
        feasible=False,
        peak_after_dbtp=peak_after,
        safe_gain_db=safe_gain,
        ceiling_dbtp=ceiling_dbtp,
        target_lufs=target_lufs,
        required_limiting=True,
        reason=(
            f"Target {target_lufs:.1f} LUFS needs {requested:+.1f} dB, but the "
            f"recording peaks at {measured.true_peak_dbtp:.1f} dBTP, which would "
            f"reach {peak_after:.1f} dBTP. The {ceiling_dbtp:.1f} dBTP ceiling allows "
            f"at most {safe_gain:+.1f} dB. Use less gain, or explicitly allow a "
            f"limiter to reach the target."
        ),
    )


def plan_shared_gain(
    measured: LoudnessResult, target_lufs: float, ceiling_dbtp: float
) -> GainPlan:
    """Plan one gain for the whole recording."""
    return _plan(measured, target_lufs, ceiling_dbtp, "shared")


def plan_per_track_gain(
    measured: LoudnessResult, target_lufs: float, ceiling_dbtp: float
) -> GainPlan:
    """Plan a gain for one track, applied to that track alone."""
    return _plan(measured, target_lufs, ceiling_dbtp, "per_track")
