"""Tests for the parts of the interface that can be checked without a screen.

The screenshots and the real clicking live in ``scripts/ui_exercise.py``;
these tests guard the things that are easy to break by accident: the
readability of the two themes, the way the theme choice is remembered, and
the words the interface uses instead of a percentage it cannot stand behind.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from splytnyt.core.model import (  # noqa: E402
    BOUNDARY_BASIS_ASSESSED,
    BOUNDARY_BASIS_MANUAL,
    BOUNDARY_BASIS_NONE,
    Track,
)
from splytnyt.ui import theme  # noqa: E402


# ----------------------------------------------------------------------
# The themes
# ----------------------------------------------------------------------
@pytest.mark.parametrize("palette", [theme.LIGHT, theme.DARK], ids=["light", "dark"])
def test_every_text_colour_is_readable_on_its_background(palette):
    """No important text may be too close to the colour behind it."""
    problems = theme.check_contrast(palette)
    assert not problems, (
        "these pairs do not reach the 4.5:1 that ordinary text needs: "
        + "; ".join(f"{name} ({ratio:.2f}:1)" for name, ratio in problems)
    )


@pytest.mark.parametrize("palette", [theme.LIGHT, theme.DARK], ids=["light", "dark"])
def test_the_body_text_is_strongly_readable(palette):
    ratio = theme.contrast_ratio(palette.text, palette.background)
    assert ratio >= 7.0, f"the body text only reaches {ratio:.1f}:1"


@pytest.mark.parametrize("palette", [theme.LIGHT, theme.DARK], ids=["light", "dark"])
def test_the_two_themes_are_not_the_same(palette):
    """A dark theme and a light theme have to actually differ."""
    other = theme.DARK if palette is theme.LIGHT else theme.LIGHT
    assert palette.background != other.background
    assert palette.text != other.text


def test_following_the_system_gives_one_of_the_two_themes():
    resolved = theme.palette_for_mode(theme.THEME_SYSTEM)
    assert resolved is theme.LIGHT or resolved is theme.DARK


def test_an_unknown_theme_name_falls_back_to_following_the_system():
    assert theme.palette_for_mode("chartreuse") in (theme.LIGHT, theme.DARK)


def test_the_theme_choice_is_remembered(tmp_path, monkeypatch):
    """The preference survives between runs."""
    # Native QSettings uses the Windows registry, ignoring APPDATA. Isolate
    # this persistence test from the user's preferences and registry access.
    from PySide6.QtCore import QSettings
    monkeypatch.setattr(theme, "QSettings", lambda *_: QSettings(
        str(tmp_path / "preferences.ini"), QSettings.IniFormat))

    theme.store_mode(theme.THEME_DARK)
    assert theme.stored_mode() == theme.THEME_DARK
    theme.store_mode(theme.THEME_LIGHT)
    assert theme.stored_mode() == theme.THEME_LIGHT
    theme.store_mode(theme.THEME_SYSTEM)
    assert theme.stored_mode() == theme.THEME_SYSTEM


def test_the_style_sheet_covers_the_widgets_that_matter():
    """The themes have to reach every kind of control, not just the canvas."""
    for palette in (theme.LIGHT, theme.DARK):
        sheet = theme.build_stylesheet(palette)
        for selector in (
            "QMainWindow", "QMenuBar", "QMenu", "QToolBar", "QPushButton",
            "QToolButton", "QLineEdit", "QComboBox", "QSpinBox", "QTableView",
            "QHeaderView", "QScrollBar", "QDockWidget", "QTabWidget",
            "QGroupBox", "QStatusBar", "QCheckBox", "QRadioButton",
            "QProgressBar", "QToolTip", "QTextEdit",
        ):
            assert selector in sheet, (
                f"the {palette.name} theme does not style {selector}"
            )


def test_a_disabled_control_is_visibly_different():
    """Disabled controls must not look like usable ones."""
    for palette in (theme.LIGHT, theme.DARK):
        sheet = theme.build_stylesheet(palette)
        assert ":disabled" in sheet, (
            f"the {palette.name} theme does not say what a disabled control "
            f"looks like"
        )


def test_the_waveform_colour_stands_out_from_the_background():
    for palette in (theme.LIGHT, theme.DARK):
        assert theme.contrast_ratio(palette.waveform, palette.background) >= 3.0, (
            f"the waveform is hard to see on the {palette.name} background"
        )


def test_the_playhead_and_the_selected_boundary_can_be_told_apart():
    """The two markers are drawn differently on purpose."""
    for palette in (theme.LIGHT, theme.DARK):
        assert palette.playhead != palette.boundary_selected
        assert palette.boundary != palette.boundary_selected
        assert palette.boundary_manual != palette.boundary_selected


# ----------------------------------------------------------------------
# The words instead of the numbers
# ----------------------------------------------------------------------
def test_a_boundary_that_was_never_assessed_says_so():
    track = Track(start_frame=0, end_frame=100, title="")
    assert track.boundary_assessment() == "Not assessed"
    assert track.identity_assessment() == "Not assessed"


def test_a_boundary_a_human_moved_is_called_manual():
    track = Track(start_frame=0, end_frame=100, title="",
                  boundary_basis=BOUNDARY_BASIS_MANUAL, boundary_confidence=0.0)
    assert track.boundary_assessment() == "Manual"


def test_an_uncertain_boundary_asks_for_a_human_decision():
    track = Track(start_frame=0, end_frame=100, title="",
                  boundary_basis=BOUNDARY_BASIS_ASSESSED,
                  boundary_confidence=0.9, boundary_uncertain=True)
    assert track.boundary_assessment() == "Needs review"


def test_an_assessed_boundary_is_described_in_words():
    track = Track(start_frame=0, end_frame=100, title="",
                  boundary_basis=BOUNDARY_BASIS_ASSESSED, boundary_confidence=0.8)
    assert track.boundary_assessment() == "High"
    track.boundary_confidence = 0.5
    assert track.boundary_assessment() == "Fair"
    track.boundary_confidence = 0.1
    assert track.boundary_assessment() == "Low"


def test_no_assessment_is_ever_a_percentage():
    """The interface must not show a precision it does not have."""
    for basis in (BOUNDARY_BASIS_NONE, BOUNDARY_BASIS_ASSESSED, BOUNDARY_BASIS_MANUAL):
        for confidence in (0.0, 0.25, 0.5, 0.75, 1.0):
            track = Track(start_frame=0, end_frame=100, title="",
                          boundary_basis=basis, boundary_confidence=confidence,
                          identity_basis=basis, identity_confidence=confidence)
            for word in (track.boundary_assessment(), track.identity_assessment()):
                assert "%" not in word
                assert not word.replace(".", "").isdigit(), (
                    f"the assessment '{word}' is a number, not a word"
                )


def test_a_pasted_title_is_not_an_acoustically_verified_identity():
    """A typed or aligned title says so; it is not "verified"."""
    typed = Track(start_frame=0, end_frame=100, title="Song",
                  identity_basis=BOUNDARY_BASIS_MANUAL, identity_confidence=1.0)
    assert typed.identity_assessment() == "Manual"

    aligned = Track(start_frame=0, end_frame=100, title="Song",
                    identity_basis=BOUNDARY_BASIS_ASSESSED, identity_confidence=1.0)
    assert aligned.identity_assessment() == "High"
    # However well the tracklist lines up, it is not an acoustic match.
    for word in (typed.identity_assessment(), aligned.identity_assessment()):
        assert "verified" not in word.lower()
        assert "acoustic" not in word.lower()


def test_the_colour_of_an_assessment_matches_the_words():
    assert theme.assessment_color("High").name() != theme.assessment_color("Low").name()
    assert theme.assessment_color("Needs review").name() != \
        theme.assessment_color("Manual").name()
