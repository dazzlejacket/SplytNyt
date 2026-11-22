"""The visual theme.

SplytNyt ships three themes: **Light**, **Dark** and **Follow System**. The
choice is remembered between sessions and takes effect immediately.

Everything the interface draws goes through this module: the waveform, the
track table, the menus, the dialogs, the tooltips, the inputs and the
disabled states. A theme is therefore a complete set of colours rather than
a handful of accents, and both themes are checked for contrast against
their own backgrounds.

The colours that carry *meaning* are the same idea in both themes - music is
green, applause is amber, an uncertain boundary is red - so a colour never
changes its meaning when the theme changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QFont, QPalette

#: The theme names the interface offers.
THEME_LIGHT = "light"
THEME_DARK = "dark"
THEME_SYSTEM = "system"

THEME_LABELS = {
    THEME_LIGHT: "Light",
    THEME_DARK: "Dark",
    THEME_SYSTEM: "Follow System",
}

SETTINGS_ORGANISATION = "SplytNyt"
SETTINGS_APPLICATION = "SplytNyt"


@dataclass(frozen=True)
class Palette:
    """Every colour the interface uses."""

    name: str
    is_dark: bool

    # Surfaces
    background: str
    surface: str
    surface_alt: str
    surface_raised: str
    border: str
    border_strong: str

    # Text
    text: str
    text_dim: str
    text_faint: str
    text_disabled: str
    text_on_accent: str

    # Accent
    accent: str
    accent_hover: str
    accent_dim: str

    # Waveform
    waveform: str
    waveform_selected: str
    ruler: str
    ruler_text: str

    # Selection
    selection: str
    selection_border: str
    selection_text: str
    alternating: str

    # Meaning
    music: str
    applause: str
    speech: str
    silence: str
    uncertain: str
    unknown: str
    locked: str
    good: str
    fair: str
    poor: str

    # Playhead and boundary
    playhead: str
    boundary: str
    boundary_selected: str
    boundary_manual: str
    #: Where the music was detected to begin. It is drawn differently from
    #: the cut, because the distance between the two is the applied margin.
    onset_marker: str

    # Region fills
    region_music: str
    region_other: str
    region_selected: str
    gap_fill: str


LIGHT = Palette(
    name="light",
    is_dark=False,
    background="#f4f6f9",
    surface="#ffffff",
    surface_alt="#eef1f6",
    surface_raised="#ffffff",
    border="#d8dde6",
    border_strong="#bcc4d1",
    text="#1b1f27",
    text_dim="#565f6e",
    text_faint="#68717d",
    text_disabled="#a3abba",
    text_on_accent="#ffffff",
    accent="#2f6fdd",
    accent_hover="#3d7ae8",
    accent_dim="#c9dcfb",
    waveform="#5d6675",
    waveform_selected="#2f6fdd",
    ruler="#ffffff",
    ruler_text="#6b7484",
    selection="#d8e6fb",
    selection_border="#2f6fdd",
    selection_text="#10233f",
    alternating="#fafbfd",
    music="#1f8a63",
    applause="#b5761a",
    speech="#7a4fb0",
    silence="#7b8494",
    uncertain="#c04a35",
    unknown="#8a93a2",
    locked="#6d4fc4",
    good="#1f7a4a",
    fair="#a9741a",
    poor="#b0432e",
    playhead="#1b1f27",
    boundary="#2f6fdd",
    boundary_selected="#0f4fae",
    boundary_manual="#7a4fb0",
    onset_marker="#b0641a",
    region_music="#dff0e8",
    region_other="#eef1f6",
    region_selected="#d5e4fb",
    gap_fill="#e7ebf1",
)

DARK = Palette(
    name="dark",
    is_dark=True,
    background="#14171c",
    surface="#1b1f26",
    surface_alt="#232833",
    surface_raised="#262c38",
    border="#2f3644",
    border_strong="#3d4655",
    text="#e7eaf0",
    text_dim="#9aa4b4",
    text_faint="#7a8392",
    text_disabled="#5a6373",
    text_on_accent="#ffffff",
    accent="#4c8dff",
    accent_hover="#5f9bff",
    accent_dim="#26456f",
    waveform="#9aa4b4",
    waveform_selected="#7fb0ff",
    ruler="#1b1f26",
    ruler_text="#8b95a6",
    selection="#24354f",
    selection_border="#4c8dff",
    selection_text="#eef3ff",
    alternating="#191d24",
    music="#3fbb8b",
    applause="#d9a441",
    speech="#c08ae8",
    silence="#6b7484",
    uncertain="#e0685f",
    unknown="#79839a",
    locked="#a78bfa",
    good="#41c98a",
    fair="#d9a441",
    poor="#e0685f",
    playhead="#f2f4f8",
    boundary="#4c8dff",
    boundary_selected="#8fbaff",
    boundary_manual="#c08ae8",
    onset_marker="#e0a35c",
    region_music="#1b332b",
    region_other="#1e232c",
    region_selected="#22334d",
    gap_fill="#191d24",
)

PALETTES = {THEME_LIGHT: LIGHT, THEME_DARK: DARK}

_active: Palette = DARK


# ----------------------------------------------------------------------
# The active palette
# ----------------------------------------------------------------------
def active() -> Palette:
    """The palette currently in use."""
    return _active


def set_active(palette: Palette) -> None:
    """Make ``palette`` the one the interface draws with."""
    global _active
    _active = palette


def system_prefers_dark() -> bool:
    """True when the desktop asks for a dark appearance."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return False
    hints = app.styleHints()
    scheme = getattr(hints, "colorScheme", None)
    if scheme is not None:
        try:
            value = scheme()
            # Qt.ColorScheme.Dark is 2, Light is 1, Unknown is 0.
            if value is not None and int(value) == 2:
                return True
            if value is not None and int(value) == 1:
                return False
        except Exception:
            pass
    window = app.palette().color(QPalette.Window)
    text = app.palette().color(QPalette.WindowText)
    return window.lightness() < text.lightness()


def palette_for_mode(mode: str) -> Palette:
    """The palette a theme name resolves to."""
    mode = (mode or THEME_SYSTEM).lower()
    if mode == THEME_LIGHT:
        return LIGHT
    if mode == THEME_DARK:
        return DARK
    return DARK if system_prefers_dark() else LIGHT


def stored_mode() -> str:
    """The theme the user last chose."""
    settings = QSettings(SETTINGS_ORGANISATION, SETTINGS_APPLICATION)
    value = settings.value("theme", THEME_SYSTEM)
    value = str(value or THEME_SYSTEM).lower()
    return value if value in (THEME_LIGHT, THEME_DARK, THEME_SYSTEM) else THEME_SYSTEM


def store_mode(mode: str) -> None:
    """Remember the theme the user chose."""
    settings = QSettings(SETTINGS_ORGANISATION, SETTINGS_APPLICATION)
    settings.setValue("theme", mode)


# ----------------------------------------------------------------------
# Colours that carry meaning
# ----------------------------------------------------------------------
def color_for_confidence(value: float, palette: Optional[Palette] = None) -> QColor:
    """Colour a confidence value: red -> amber -> green."""
    palette = palette or _active
    value = max(0.0, min(1.0, value))
    if value < 0.5:
        ratio = value / 0.5
        low, high = _rgb(palette.poor), _rgb(palette.fair)
    else:
        ratio = (value - 0.5) / 0.5
        low, high = _rgb(palette.fair), _rgb(palette.good)
    return QColor(
        int(low[0] + (high[0] - low[0]) * ratio),
        int(low[1] + (high[1] - low[1]) * ratio),
        int(low[2] + (high[2] - low[2]) * ratio),
    )


def _rgb(value: str) -> tuple[int, int, int]:
    colour = QColor(value)
    return colour.red(), colour.green(), colour.blue()


def relative_luminance(colour: str) -> float:
    """The relative luminance of a colour, as WCAG defines it."""
    channels = []
    for component in _rgb(colour):
        value = component / 255.0
        channels.append(
            value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        )
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(foreground: str, background: str) -> float:
    """The contrast ratio of two colours, from 1.0 to 21.0.

    4.5 is the ratio that ordinary text has to reach to be readable, and
    3.0 is enough for large text and for the edges of controls.
    """
    first = relative_luminance(foreground)
    second = relative_luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def check_contrast(palette: "Palette") -> list[tuple[str, float]]:
    """The pairs in a palette that are too close together to read.

    An empty list means every important pair in the palette reaches the
    4.5:1 that ordinary text needs.
    """
    pairs = [
        ("text on background", palette.text, palette.background),
        ("text on surface", palette.text, palette.surface),
        ("text on raised surface", palette.text, palette.surface_raised),
        ("dimmed text on background", palette.text_dim, palette.background),
        ("faint text on background", palette.text_faint, palette.background),
        ("text on selection", palette.selection_text, palette.selection),
    ]
    return [
        (name, contrast_ratio(foreground, background))
        for name, foreground, background in pairs
        if contrast_ratio(foreground, background) < 4.5
    ]


def color_for_section(section_type: str, palette: Optional[Palette] = None) -> QColor:
    """The colour a section type is drawn in."""
    palette = palette or _active
    return QColor(
        {
            "music": palette.music,
            "applause": palette.applause,
            "speech": palette.speech,
            "dialogue": palette.speech,
            "silence": palette.silence,
            "uncertain": palette.uncertain,
            "unknown": palette.unknown,
        }.get(section_type, palette.unknown)
    )


def assessment_color(text: str, palette: Optional[Palette] = None) -> QColor:
    """The colour a written assessment is drawn in."""
    palette = palette or _active
    lowered = (text or "").lower()
    if lowered in ("high", "confirmed", "manual", "typed"):
        return QColor(palette.good if lowered in ("high", "confirmed") else palette.locked)
    if lowered in ("fair", "likely", "aligned"):
        return QColor(palette.fair)
    if lowered in ("low", "needs review", "conflict", "uncertain"):
        return QColor(palette.uncertain if lowered in ("needs review", "conflict", "uncertain") else palette.poor)
    return QColor(palette.text_dim)


# ----------------------------------------------------------------------
# Fonts
# ----------------------------------------------------------------------
def base_font() -> QFont:
    font = QFont()
    font.setPointSize(9)
    return font


def mono_font() -> QFont:
    font = QFont("Consolas" if _has_consolas() else "Monospace")
    font.setPointSize(9)
    return font


def _has_consolas() -> bool:
    from PySide6.QtGui import QFontDatabase

    return "Consolas" in QFontDatabase.families()


# ----------------------------------------------------------------------
# The stylesheet
# ----------------------------------------------------------------------
def build_stylesheet(palette: Palette) -> str:
    """The complete stylesheet for one palette."""
    return f"""
QWidget {{
    background-color: {palette.background};
    color: {palette.text};
}}
QMainWindow, QDialog {{
    background-color: {palette.background};
}}

/* --- menus and toolbars --- */
QMenuBar {{
    background-color: {palette.surface};
    color: {palette.text};
    padding: 1px;
    border-bottom: 1px solid {palette.border};
}}
QMenuBar::item {{ padding: 5px 11px; background: transparent; border-radius: 4px; }}
QMenuBar::item:selected {{ background: {palette.surface_alt}; }}
QMenuBar::item:pressed {{ background: {palette.surface_alt}; }}

QMenu {{
    background-color: {palette.surface};
    color: {palette.text};
    border: 1px solid {palette.border};
    padding: 5px;
}}
QMenu::item {{ padding: 6px 28px 6px 24px; border-radius: 4px; }}
QMenu::item:selected {{ background-color: {palette.selection}; color: {palette.selection_text}; }}
QMenu::item:disabled {{ color: {palette.text_disabled}; }}
QMenu::separator {{ height: 1px; background: {palette.border}; margin: 5px 10px; }}

QToolBar {{
    background-color: {palette.surface};
    border: 0px;
    border-bottom: 1px solid {palette.border};
    spacing: 2px;
    padding: 5px 8px;
}}
QToolBar::separator {{
    width: 1px;
    background: {palette.border};
    margin: 4px 8px;
}}
QToolButton {{
    color: {palette.text};
    background: transparent;
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 5px 11px;
    margin: 1px;
}}
QToolButton:hover {{ background: {palette.surface_alt}; border-color: {palette.border}; }}
QToolButton:pressed {{ background: {palette.accent_dim}; }}
QToolButton:checked {{
    background: {palette.selection};
    border-color: {palette.accent};
    color: {palette.selection_text};
}}
QToolButton:disabled {{ color: {palette.text_disabled}; }}
QToolButton::menu-indicator {{ image: none; width: 0px; }}

/* --- buttons --- */
QPushButton {{
    background-color: {palette.surface_alt};
    border: 1px solid {palette.border};
    border-radius: 5px;
    padding: 6px 14px;
    color: {palette.text};
    min-height: 18px;
}}
QPushButton:hover {{ background-color: {palette.surface_raised}; border-color: {palette.border_strong}; }}
QPushButton:pressed {{ background-color: {palette.accent_dim}; }}
QPushButton:focus {{ border-color: {palette.accent}; outline: none; }}
QPushButton:disabled {{ color: {palette.text_disabled}; border-color: {palette.border}; background: {palette.background}; }}
QPushButton[primary="true"] {{
    background-color: {palette.accent};
    border-color: {palette.accent};
    color: {palette.text_on_accent};
    font-weight: 600;
}}
QPushButton[primary="true"]:hover {{ background-color: {palette.accent_hover}; border-color: {palette.accent_hover}; }}
QPushButton[primary="true"]:disabled {{ background-color: {palette.surface_alt}; border-color: {palette.border}; color: {palette.text_disabled}; }}

/* --- inputs --- */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {palette.surface};
    border: 1px solid {palette.border};
    border-radius: 5px;
    padding: 5px 7px;
    color: {palette.text};
    selection-background-color: {palette.accent};
    selection-color: {palette.text_on_accent};
    min-height: 18px;
}}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{ border-color: {palette.border_strong}; }}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {palette.accent};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {palette.text_disabled};
    background-color: {palette.background};
}}
QLineEdit::placeholder {{ color: {palette.text_faint}; }}
QComboBox::drop-down {{ border: 0px; width: 20px; }}
QComboBox::down-arrow {{ image: none; }}
QComboBox QAbstractItemView {{
    background-color: {palette.surface};
    border: 1px solid {palette.border};
    selection-background-color: {palette.selection};
    selection-color: {palette.selection_text};
    outline: none;
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 16px; border: 0px; }}

/* --- tables and lists --- */
QTableView, QTreeView, QListView {{
    background-color: {palette.surface};
    alternate-background-color: {palette.alternating};
    gridline-color: {palette.border};
    border: 1px solid {palette.border};
    border-radius: 6px;
    selection-background-color: {palette.selection};
    selection-color: {palette.selection_text};
    outline: none;
}}
QTableView::item {{ padding: 3px 6px; border: 0px; }}
QTableView::item:selected {{ background-color: {palette.selection}; color: {palette.selection_text}; }}
QTableView::item:focus {{ outline: none; }}
QHeaderView {{ background-color: {palette.surface}; }}
QHeaderView::section {{
    background-color: {palette.surface};
    color: {palette.text_dim};
    border: 0px;
    border-right: 1px solid {palette.border};
    border-bottom: 1px solid {palette.border};
    padding: 6px 8px;
    font-weight: 600;
}}
QHeaderView::section:hover {{ background-color: {palette.surface_alt}; color: {palette.text}; }}
QTableCornerButton::section {{ background-color: {palette.surface}; border: 0px; }}

/* --- scrollbars --- */
QScrollBar:vertical {{ background: transparent; width: 12px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 2px; }}
QScrollBar::handle {{ background: {palette.border_strong}; border-radius: 5px; min-height: 28px; min-width: 28px; }}
QScrollBar::handle:hover {{ background: {palette.text_faint}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0px; width: 0px; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

/* --- splitters, docks, tabs --- */
QSplitter::handle {{ background: {palette.border}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QDockWidget {{ titlebar-close-icon: none; }}
QDockWidget::title {{
    background: {palette.surface};
    padding: 7px 10px;
    border-bottom: 1px solid {palette.border};
    color: {palette.text_dim};
    font-weight: 600;
}}
QTabWidget::pane {{ border: 1px solid {palette.border}; border-radius: 6px; top: -1px; }}
QTabBar::tab {{
    background: transparent;
    color: {palette.text_dim};
    padding: 7px 15px;
    border: 0px;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{ color: {palette.text}; border-bottom: 2px solid {palette.accent}; }}
QTabBar::tab:hover {{ color: {palette.text}; }}

/* --- group boxes --- */
QGroupBox {{
    border: 1px solid {palette.border};
    border-radius: 7px;
    margin-top: 14px;
    padding: 10px 10px 8px 10px;
    background-color: {palette.surface};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 11px;
    padding: 0px 5px;
    color: {palette.text_dim};
    font-weight: 600;
}}

/* --- status bar --- */
QStatusBar {{
    background-color: {palette.surface};
    color: {palette.text_dim};
    border-top: 1px solid {palette.border};
}}
QStatusBar::item {{ border: 0px; }}
QStatusBar QLabel {{ color: {palette.text_dim}; padding: 2px 6px; }}

/* --- checks, radios, progress --- */
QCheckBox, QRadioButton {{ color: {palette.text}; spacing: 7px; background: transparent; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 15px; height: 15px; }}
QCheckBox::indicator {{
    border: 1px solid {palette.border_strong};
    border-radius: 4px;
    background: {palette.surface};
}}
QCheckBox::indicator:hover {{ border-color: {palette.accent}; }}
QCheckBox::indicator:checked {{ background: {palette.accent}; border-color: {palette.accent}; }}
QCheckBox::indicator:disabled {{ border-color: {palette.border}; background: {palette.background}; }}
QRadioButton::indicator {{
    border: 1px solid {palette.border_strong};
    border-radius: 8px;
    background: {palette.surface};
}}
QRadioButton::indicator:checked {{ background: {palette.accent}; border-color: {palette.accent}; }}

QProgressBar {{
    background: {palette.surface_alt};
    border: 1px solid {palette.border};
    border-radius: 5px;
    text-align: center;
    color: {palette.text};
    height: 18px;
}}
QProgressBar::chunk {{ background: {palette.accent}; border-radius: 4px; }}

/* --- labels --- */
QLabel {{ color: {palette.text}; background: transparent; }}
QLabel[dim="true"] {{ color: {palette.text_dim}; }}
QLabel[hint="true"] {{ color: {palette.text_faint}; }}
QLabel[heading="true"] {{ color: {palette.text}; font-weight: 600; }}

/* --- tooltips --- */
QToolTip {{
    background-color: {palette.surface_raised};
    color: {palette.text};
    border: 1px solid {palette.border_strong};
    border-radius: 5px;
    padding: 6px 8px;
}}

/* --- menus in widgets --- */
QMenuBar QToolTip, QToolBar QToolTip {{ border: 1px solid {palette.border_strong}; }}
"""


def apply_theme(app, mode: str = THEME_SYSTEM) -> Palette:
    """Apply a theme to the application and return the palette in use."""
    palette = palette_for_mode(mode)
    set_active(palette)
    app.setStyleSheet(build_stylesheet(palette))

    colours = QPalette()
    colours.setColor(QPalette.Window, QColor(palette.background))
    colours.setColor(QPalette.WindowText, QColor(palette.text))
    colours.setColor(QPalette.Base, QColor(palette.surface))
    colours.setColor(QPalette.AlternateBase, QColor(palette.alternating))
    colours.setColor(QPalette.Text, QColor(palette.text))
    colours.setColor(QPalette.PlaceholderText, QColor(palette.text_faint))
    colours.setColor(QPalette.Button, QColor(palette.surface_alt))
    colours.setColor(QPalette.ButtonText, QColor(palette.text))
    colours.setColor(QPalette.Highlight, QColor(palette.selection))
    colours.setColor(QPalette.HighlightedText, QColor(palette.selection_text))
    colours.setColor(QPalette.ToolTipBase, QColor(palette.surface_raised))
    colours.setColor(QPalette.ToolTipText, QColor(palette.text))
    colours.setColor(QPalette.Light, QColor(palette.border))
    colours.setColor(QPalette.Midlight, QColor(palette.border))
    colours.setColor(QPalette.Dark, QColor(palette.border))
    colours.setColor(QPalette.Mid, QColor(palette.border))
    colours.setColor(QPalette.Shadow, QColor(palette.border_strong))
    colours.setColor(QPalette.Link, QColor(palette.accent))
    colours.setColor(QPalette.Disabled, QPalette.Text, QColor(palette.text_disabled))
    colours.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(palette.text_disabled))
    colours.setColor(QPalette.Disabled, QPalette.WindowText, QColor(palette.text_disabled))
    colours.setColor(QPalette.Disabled, QPalette.Text, QColor(palette.text_disabled))
    app.setPalette(colours)
    app.setFont(base_font())
    return palette
