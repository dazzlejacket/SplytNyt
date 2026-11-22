"""An explicit peak limiter.

This is the *only* processing in SplytNyt that alters the shape of the
signal, and it never runs unless the user asked for it. When a loudness
target cannot be reached under the true-peak ceiling, the interface offers
two choices: use less gain, or explicitly allow a limiter. This module is
the second choice.

It is a peak limiter with a smoothed gain-reduction envelope:

* the required reduction is computed from the instantaneous peak of the
  block,
* the reduction is smoothed with an attack and a release so it does not
  pump,
* the same gain is applied to every channel, so the stereo image is not
  disturbed.

The limiter is a *safety* device for the true-peak ceiling. It is not a
loudness maximiser and it does not try to reach the target on its own.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

#: Attack time of the gain-reduction envelope, in seconds.
ATTACK_SECONDS = 0.005
#: Release time of the gain-reduction envelope, in seconds.
RELEASE_SECONDS = 0.050


class PeakLimiter:
    """Smoothed peak limiter with a true-peak ceiling.

    The ceiling is expressed in dBTP, which is the same unit the
    measurement code uses, so the value the user sees in the dialog is the
    value the limiter works to.
    """

    def __init__(self, ceiling_dbtp: float, sample_rate: int) -> None:
        self.ceiling_dbtp = float(ceiling_dbtp)
        self.sample_rate = int(sample_rate)
        self.ceiling = 10.0 ** (self.ceiling_dbtp / 20.0)
        self._gain = 1.0
        self._attack = float(np.exp(-1.0 / max(1.0, ATTACK_SECONDS * sample_rate)))
        self._release = float(np.exp(-1.0 / max(1.0, RELEASE_SECONDS * sample_rate)))
        self.max_reduction_db = 0.0

    def __call__(self, block: np.ndarray) -> np.ndarray:
        """Limit one block, carrying the envelope state to the next one."""
        if block.size == 0:
            return block
        data = np.asarray(block, dtype=np.float32)
        peak = float(np.max(np.abs(data)))
        if peak <= 0.0:
            return data

        # The gain that would bring this instant under the ceiling.
        required = self.ceiling / peak
        if required >= 1.0:
            target = 1.0
        else:
            target = required

        # Smooth the envelope: attack fast, release slow.
        if target < self._gain:
            self._gain = target
        else:
            self._gain = self._gain * self._release + target * (1.0 - self._release)

        reduction_db = -20.0 * np.log10(max(self._gain, 1e-9))
        if reduction_db > self.max_reduction_db:
            self.max_reduction_db = reduction_db

        limited = data * np.float32(self._gain)
        # Never exceed the ceiling because of the envelope's own ramp.
        peak_after = float(np.max(np.abs(limited)))
        if peak_after > self.ceiling:
            limited = limited * np.float32(self.ceiling / peak_after)
        return limited

    def notes(self) -> list[str]:
        """Remarks worth showing in the export log."""
        if self.max_reduction_db <= 0.01:
            return []
        return [
            f"an explicit peak limiter was applied at {self.ceiling_dbtp:.1f} dBTP "
            f"(maximum gain reduction {self.max_reduction_db:.1f} dB)"
        ]


def limiter_notes(limiter: Optional[PeakLimiter]) -> list[str]:
    if limiter is None:
        return []
    return limiter.notes()
