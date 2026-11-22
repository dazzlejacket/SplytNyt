"""The waveform view.

A custom ``QAbstractScrollArea`` that draws:

* a time ruler,
* the waveform, one lane per channel, from a multi-resolution peak pyramid,
* the track regions, tinted by their section type and labelled with the
  track number and name,
* the boundary markers, which can be dragged to adjust a cut,
* the playhead and the cursor, drawn so they cannot be confused with each
  other or with the selected boundary.

Zooming is done in frames-per-pixel, so a boundary keeps its exact sample
position at every zoom level: the display is a view, never the source of
truth.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QAbstractScrollArea, QMenu, QToolTip, QWidget

from ..audio.peaks import PeakPyramid
from ..core.model import Track
from ..core.units import frames_to_timestamp
from . import theme

OVERVIEW_HEIGHT = 22
RULER_HEIGHT = 28
HEADER_HEIGHT = OVERVIEW_HEIGHT + RULER_HEIGHT
MIN_FRAMES_PER_PIXEL = 1.0
MAX_ZOOM_OUT = 2000.0
#: Height of the coloured band that shows the section type of a track.
REGION_BAND = 5


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class WaveformView(QAbstractScrollArea):
    """Zoomable waveform with draggable shared track dividers and overview bar."""

    cursorMoved = Signal(int)
    boundaryDragged = Signal(int, int)     # old frame, new frame
    boundaryClicked = Signal(int)          # frame
    trackSelected = Signal(str)            # track id
    zoomChanged = Signal(float)
    seekRequested = Signal(int)            # double-click seek frame
    splitRequested = Signal(int)           # context-menu split frame
    removeSplitRequested = Signal(int)     # context-menu remove divider frame
    renameTrackRequested = Signal(str)     # context-menu rename track id
    playTrackRequested = Signal(str)       # context-menu play track id
    zoomTrackRequested = Signal(str)       # context-menu zoom to track id
    toggleLockRequested = Signal(str)      # context-menu lock/unlock track id

    #: What the empty state says, when there is no audio yet.
    EMPTY_STATE_LINES = (
        "No audio yet",
        "Import a recording with File > Import audio, or drop one onto this window.",
        "SplytNyt reads your files without ever changing them.",
    )

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumHeight(200)
        self.setFrameShape(QAbstractScrollArea.NoFrame)

        self._sources: list = []
        self._source_starts: list[int] = []
        self._pyramids: dict[str, PeakPyramid] = {}
        self._tracks: list[Track] = []
        self._channels = 2
        self._total_frames = 0
        self._sample_rate = 44100

        self._frames_per_pixel = 100.0
        self._cursor = 0
        self._playhead = 0
        self._selected_track_id = ""
        self._selected_boundary: Optional[int] = None
        self._onset_frame: Optional[int] = None
        self._drag_origin: Optional[int] = None
        self._drag_frame: Optional[int] = None
        self._drag_bounds: Optional[tuple[int, int]] = None
        self._dragging_overview: bool = False
        self._hover_boundary: Optional[int] = None
        self._reasons: dict[int, list[str]] = {}
        self._track_numbers: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def set_project(self, project, pyramids: Optional[dict[str, PeakPyramid]] = None) -> None:
        self._sources = list(project.sources)
        self._channels = project.channels() or 2
        self._sample_rate = project.sample_rate() or 44100
        self._total_frames = project.total_frames()
        if pyramids is not None:
            self._pyramids = pyramids
        self._update_scroll_range()
        self.viewport().update()

    def set_pyramids(self, pyramids: dict[str, PeakPyramid]) -> None:
        self._pyramids = pyramids
        self.viewport().update()

    def set_tracks(self, tracks: list[Track]) -> None:
        self._tracks = sorted(tracks, key=lambda t: t.start_frame)
        self._reasons = {}
        self._track_numbers = {}
        for index, track in enumerate(self._tracks, start=1):
            self._track_numbers[track.track_id] = index
            for frame in (track.start_frame, track.end_frame):
                self._reasons.setdefault(frame, list(track.reasons))
        self.viewport().update()

    def set_playhead(self, frame: int) -> None:
        self._playhead = int(frame)
        self.viewport().update()

    def set_cursor(self, frame: int) -> None:
        self._cursor = int(frame)
        self.viewport().update()

    @property
    def cursor_frame(self) -> int:
        return self._cursor

    @property
    def selected_boundary(self) -> Optional[int]:
        return self._selected_boundary

    def select_track(self, track_id: str) -> None:
        self._selected_track_id = track_id
        self.viewport().update()

    def set_selected_boundary(self, frame: Optional[int]) -> None:
        """Highlight one boundary as the one being reviewed."""
        self._selected_boundary = None if frame is None else int(frame)
        self.viewport().update()

    def set_onset_marker(self, frame: Optional[int]) -> None:
        """Show where the music was detected to begin for the reviewed cut.

        This is a different place from the cut itself: the difference
        between the two is the margin that was applied.
        """
        self._onset_frame = None if frame is None else int(frame)
        self.viewport().update()

    @property
    def onset_frame(self) -> Optional[int]:
        return self._onset_frame

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------
    @property
    def frames_per_pixel(self) -> float:
        return self._frames_per_pixel

    def zoom_to_fit(self) -> None:
        width = max(1, self.viewport().width())
        if self._total_frames <= 0:
            self._frames_per_pixel = 100.0
        else:
            self._frames_per_pixel = max(MIN_FRAMES_PER_PIXEL, self._total_frames / width)
        self._update_scroll_range()
        self.zoomChanged.emit(self._frames_per_pixel)
        self.viewport().update()

    def zoom_to_frame(self, frame: int, span_seconds: float = 12.0) -> None:
        """Centre the view on a frame and zoom to a window around it.

        This is what "zoom to boundary" uses: the cut has to be judged
        with a few seconds of context on either side.
        """
        if self._total_frames <= 0:
            return
        span_frames = max(1, int(span_seconds * self._sample_rate))
        width = max(1, self.viewport().width())
        self._frames_per_pixel = max(MIN_FRAMES_PER_PIXEL, span_frames / width)
        self._update_scroll_range()
        centre_pixel = int(frame / self._frames_per_pixel)
        self.horizontalScrollBar().setValue(max(0, centre_pixel - width // 2))
        self.zoomChanged.emit(self._frames_per_pixel)
        self.viewport().update()

    def _max_frames_per_pixel(self) -> float:
        width = max(1, self.viewport().width())
        if self._total_frames <= 0:
            return MAX_ZOOM_OUT
        return max(MAX_ZOOM_OUT, self._total_frames / float(width))

    def zoom_by(self, factor: float, anchor_frame: Optional[int] = None) -> None:
        """Zoom by ``factor`` (<1 zooms in) keeping ``anchor_frame`` in place."""
        if anchor_frame is None:
            anchor_frame = self._cursor
        new_fpp = max(MIN_FRAMES_PER_PIXEL,
                      min(self._max_frames_per_pixel(), self._frames_per_pixel * factor))
        if abs(new_fpp - self._frames_per_pixel) < 1e-9:
            return
        anchor_pixel = (anchor_frame - self._offset_frame()) / self._frames_per_pixel
        self._frames_per_pixel = new_fpp
        self._update_scroll_range()
        # Re-anchor so the frame under the pointer does not move.
        target = anchor_frame / new_fpp - anchor_pixel
        self.horizontalScrollBar().setValue(int(round(target)))
        self.zoomChanged.emit(self._frames_per_pixel)
        self.viewport().update()

    def _update_scroll_range(self) -> None:
        width = max(1, self.viewport().width())
        if self._total_frames <= 0:
            self.horizontalScrollBar().setRange(0, 0)
            return
        pixels = int(self._total_frames / self._frames_per_pixel) + 1
        bar = self.horizontalScrollBar()
        bar.setRange(0, max(0, pixels - width))
        bar.setPageStep(width)
        bar.setSingleStep(50)

    def _offset_frame(self) -> int:
        return int(self.horizontalScrollBar().value() * self._frames_per_pixel)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_scroll_range()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def _frame_to_x(self, frame: int) -> float:
        return (frame - self._offset_frame()) / self._frames_per_pixel

    def _x_to_frame(self, x: float) -> int:
        return int(round(self._offset_frame() + x * self._frames_per_pixel))

    def _visible_range(self) -> tuple[int, int]:
        start = self._offset_frame()
        end = start + int(self.viewport().width() * self._frames_per_pixel)
        return start, end

    def _lane_rects(self) -> list[QRect]:
        height = max(20, self.viewport().height() - HEADER_HEIGHT)
        count = max(1, self._channels)
        lane_height = height / count
        rects = []
        for index in range(count):
            top = HEADER_HEIGHT + index * lane_height
            rects.append(QRect(0, int(top), self.viewport().width(), int(lane_height)))
        return rects

    def _effective_track_span(self, track: Track) -> tuple[int, int]:
        """Return ``(start, end)`` with any live divider drag applied."""
        start, end = track.start_frame, track.end_frame
        if self._drag_origin is not None and self._drag_frame is not None:
            if end == self._drag_origin:
                end = self._drag_frame
            if start == self._drag_origin:
                start = self._drag_frame
        return start, end

    def _adjacent_tracks_for_boundary(self, frame: int) -> tuple[Optional[Track], Optional[Track]]:
        for left, right in zip(self._tracks, self._tracks[1:]):
            if left.end_frame == frame and right.start_frame == frame:
                return left, right
        return None, None

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:
        palette = theme.active()
        painter = QPainter(self.viewport())
        try:
            painter.setRenderHint(QPainter.Antialiasing, False)
            painter.fillRect(self.viewport().rect(), QColor(palette.background))

            if self._total_frames <= 0:
                self._paint_empty_state(painter)
                return

            self._paint_tracks(painter)
            self._paint_waveform(painter)
            self._paint_boundaries(painter)
            self._paint_overview(painter)
            self._paint_ruler(painter)
            self._paint_cursor(painter)
            self._paint_drag_overlay(painter)
        finally:
            # The painter is ended here rather than left to the garbage
            # collector, so the viewport's backing store is never closed
            # while a painter is still active on it.
            painter.end()

    def _paint_empty_state(self, painter: QPainter) -> None:
        """Say what to do next instead of showing an empty box."""
        palette = theme.active()
        rect = self.viewport().rect()
        painter.setPen(QColor(palette.text))
        font = QFont(painter.font())
        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            rect.adjusted(0, -30, 0, 0),
            Qt.AlignCenter,
            self.EMPTY_STATE_LINES[0],
        )
        font.setPointSize(9)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor(palette.text_dim))
        painter.drawText(
            rect.adjusted(0, 10, 0, 0),
            Qt.AlignCenter,
            self.EMPTY_STATE_LINES[1],
        )
        painter.setPen(QColor(palette.text_faint))
        painter.drawText(
            rect.adjusted(0, 32, 0, 0),
            Qt.AlignCenter,
            self.EMPTY_STATE_LINES[2],
        )

    def _empty_state_text(self) -> str:
        """What the empty state says, as plain text.

        The same words are drawn on the canvas and offered here, so the
        check and the picture can never drift apart.
        """
        return "\n".join(self.EMPTY_STATE_LINES)

    def _paint_tracks(self, painter: QPainter) -> None:
        palette = theme.active()
        lanes = self._lane_rects()
        top = lanes[0].top()
        bottom = lanes[-1].top() + lanes[-1].height()

        for index, track in enumerate(self._tracks):
            eff_start, eff_end = self._effective_track_span(track)
            left = self._frame_to_x(eff_start)
            right = self._frame_to_x(eff_end)
            if right < 0 or left > self.viewport().width():
                continue
            selected = track.track_id == self._selected_track_id
            width = max(1.0, right - left)

            # Adjacent track blocks use alternating light tints so Track 1,
            # Track 2, Track 3 are clearly distinct blocks on one timeline.
            if selected:
                fill = QColor(palette.region_selected)
                fill.setAlpha(135)
            elif track.section_type in ("music", "musical_number"):
                fill = QColor(palette.region_music)
                fill.setAlpha(85 if index % 2 == 0 else 55)
            else:
                fill = QColor(palette.region_other)
                fill.setAlpha(85 if index % 2 == 0 else 55)

            rect = QRectF(left, top, width, bottom - top)
            painter.fillRect(rect, fill)

            # A band along the top shows the track block and its section type.
            band = QColor(theme.color_for_section(track.section_type, palette))
            band.setAlpha(245 if selected else 185)
            painter.fillRect(QRectF(left, top, width, REGION_BAND), band)

            if selected:
                painter.setPen(QPen(QColor(palette.selection_border), 1))
                painter.drawRect(rect)

            self._paint_track_label(
                painter, track, left, width, top, selected, eff_end - eff_start
            )

    def _paint_track_label(
        self,
        painter: QPainter,
        track: Track,
        left: float,
        width: float,
        top: int,
        selected: bool,
        span_frames: Optional[int] = None,
    ) -> None:
        """Put the track number, name, and duration directly on the block."""
        palette = theme.active()
        number = self._track_numbers.get(track.track_id, 0)
        name = track.title or f"Track {number}" if number else (track.title or "Untitled")
        frames = track.frames if span_frames is None else max(0, span_frames)
        duration = _format_duration(frames / float(self._sample_rate or 44100))
        lock_prefix = "* " if track.locked else ""
        font = QFont(painter.font())
        font.setPointSize(8)
        font.setBold(selected)
        painter.setFont(font)
        metrics = QFontMetrics(font)

        # Clip text region cleanly to visible portion of the block.
        vis_left = max(0.0, left)
        vis_right = min(float(self.viewport().width()), left + width)
        vis_width = vis_right - vis_left
        if vis_width < 18:
            return

        candidates = [
            f"{lock_prefix}{number}. {name} ({duration})" if number else f"{lock_prefix}{name} ({duration})",
            f"{lock_prefix}{number}. {name}" if number else f"{lock_prefix}{name}",
            f"{lock_prefix}{number} ({duration})" if number else duration,
            f"{lock_prefix}{number}" if number else "",
        ]
        chosen = ""
        for candidate in candidates:
            if candidate and metrics.horizontalAdvance(candidate) + 12 <= vis_width:
                chosen = candidate
                break
        if not chosen and vis_width >= 44:
            chosen = metrics.elidedText(candidates[0], Qt.ElideRight, int(vis_width - 10))
        if not chosen:
            return

        painter.setPen(QColor(palette.text if selected else palette.text_dim))
        painter.drawText(
            QRectF(vis_left + 6, top + REGION_BAND + 3, vis_width - 10, 16),
            Qt.AlignLeft | Qt.AlignVCenter,
            chosen,
        )

    def _paint_waveform(self, painter: QPainter) -> None:
        palette = theme.active()
        start, end = self._visible_range()
        width = self.viewport().width()
        if end <= start:
            return
        buckets = max(1, width)
        mins, maxs = self._span_peaks(start, end, buckets)
        lanes = self._lane_rects()
        channel_count = min(self._channels, mins.shape[1]) if mins.size else 0

        for channel in range(channel_count):
            lane = lanes[channel]
            centre = lane.top() + lane.height() / 2.0
            half = lane.height() / 2.0 - 3.0
            painter.setPen(Qt.NoPen)
            for x in range(buckets):
                low = float(mins[x, channel])
                high = float(maxs[x, channel])
                if low == 0.0 and high == 0.0:
                    continue
                top = centre - high * half
                bottom = centre - low * half
                if bottom - top < 1.0:
                    bottom = top + 1.0
                painter.setBrush(QBrush(QColor(palette.waveform)))
                painter.drawRect(QRectF(x, top, 1.0, bottom - top))

    def _paint_boundaries(self, painter: QPainter) -> None:
        palette = theme.active()
        lanes = self._lane_rects()
        top = lanes[0].top()
        bottom = lanes[-1].top() + lanes[-1].height()

        drawn: set[int] = set()
        for track in self._tracks:
            for frame, is_start in ((track.start_frame, True), (track.end_frame, False)):
                if frame in drawn:
                    continue
                drawn.add(frame)
                # While dragging this shared divider, draw it at its live position.
                draw_frame = (
                    self._drag_frame
                    if (self._drag_origin == frame and self._drag_frame is not None)
                    else frame
                )
                x = self._frame_to_x(draw_frame)
                if x < -6 or x > self.viewport().width() + 6:
                    continue
                is_dragging = self._drag_origin == frame and self._drag_frame is not None
                is_hovered = self._hover_boundary == frame
                is_selected = self._selected_boundary == frame or is_dragging
                if frame == 0 or frame == self._total_frames:
                    colour = QColor(palette.text_faint)
                elif is_selected or is_hovered:
                    colour = QColor(palette.boundary_selected)
                elif track.boundary_basis == "manual":
                    colour = QColor(palette.boundary_manual)
                elif track.boundary_uncertain:
                    colour = QColor(palette.uncertain)
                else:
                    colour = QColor(palette.boundary)

                pen_width = 3 if is_dragging else (2 if (is_selected or is_hovered) else 1)
                pen = QPen(colour, pen_width)
                if is_selected:
                    pen.setStyle(Qt.SolidLine)
                painter.setPen(pen)
                painter.drawLine(QPointF(x, top), QPointF(x, bottom))

                # A handle at the top of the boundary (larger when hovered/dragged).
                if 0 < frame < self._total_frames:
                    painter.setBrush(QBrush(colour))
                    painter.setPen(Qt.NoPen)
                    half_w = 6.0 if (is_hovered or is_dragging) else 4.5
                    handle_h = 9.0 if (is_hovered or is_dragging) else 7.0
                    path = QPainterPath()
                    path.moveTo(x - half_w, top)
                    path.lineTo(x + half_w, top)
                    path.lineTo(x, top + handle_h)
                    path.closeSubpath()
                    painter.drawPath(path)

                if is_selected and 0 < frame < self._total_frames:
                    painter.setBrush(QBrush(colour))
                    painter.setPen(Qt.NoPen)
                    marker = QPainterPath()
                    marker.moveTo(x - 5, bottom)
                    marker.lineTo(x + 5, bottom)
                    marker.lineTo(x, bottom - 7)
                    marker.closeSubpath()
                    painter.drawPath(marker)

        self._paint_onset_marker(painter, top, bottom)

    def _paint_onset_marker(self, painter: QPainter, top: int, bottom: int) -> None:
        """Draw where the music was detected to begin.

        The onset and the cut are two different places, and the difference
        between them is the margin that was applied. They are drawn
        differently so that the margin can be seen, not inferred.
        """
        if self._onset_frame is None or self._onset_frame <= 0:
            return
        x = self._frame_to_x(self._onset_frame)
        if x < -4 or x > self.viewport().width() + 4:
            return

        palette = theme.active()
        colour = QColor(palette.onset_marker)
        pen = QPen(colour, 1)
        pen.setStyle(Qt.DashLine)
        pen.setDashPattern([3.0, 3.0])
        painter.setPen(pen)
        painter.drawLine(QPointF(x, top), QPointF(x, bottom))

        # A small diamond at the top, so the onset can be told from the
        # triangular handles of the cut.
        painter.setPen(QPen(colour, 1))
        painter.setBrush(QBrush(colour))
        diamond = QPainterPath()
        diamond.moveTo(x, top)
        diamond.lineTo(x + 4.5, top + 5)
        diamond.lineTo(x, top + 10)
        diamond.lineTo(x - 4.5, top + 5)
        diamond.closeSubpath()
        painter.drawPath(diamond)

    def _paint_overview(self, painter: QPainter) -> None:
        """Full-recording navigator strip for multi-hour recordings."""
        palette = theme.active()
        width = max(1, self.viewport().width())
        rect = QRect(0, 0, width, OVERVIEW_HEIGHT)
        painter.fillRect(rect, QColor(palette.surface))

        if self._total_frames > 0:
            scale = width / float(self._total_frames)
            for index, track in enumerate(self._tracks):
                eff_start, eff_end = self._effective_track_span(track)
                x0 = eff_start * scale
                x1 = eff_end * scale
                w = max(1.0, x1 - x0)
                colour = QColor(theme.color_for_section(track.section_type, palette))
                if track.track_id == self._selected_track_id:
                    colour.setAlpha(210)
                else:
                    colour.setAlpha(120 if index % 2 == 0 else 80)
                painter.fillRect(QRectF(x0, 2, w, OVERVIEW_HEIGHT - 4), colour)
                if index > 0:
                    painter.setPen(QPen(QColor(palette.background), 1))
                    painter.drawLine(QPointF(x0, 2), QPointF(x0, OVERVIEW_HEIGHT - 2))

            # Current playhead in overview
            px = self._playhead * scale
            painter.setPen(QPen(QColor(palette.playhead), 1))
            painter.drawLine(QPointF(px, 1), QPointF(px, OVERVIEW_HEIGHT - 1))

            # Visible viewport indicator
            vis_start, vis_end = self._visible_range()
            vx0 = max(0.0, vis_start * scale)
            vx1 = min(float(width), vis_end * scale)
            vw = max(6.0, vx1 - vx0)
            vp_fill = QColor(palette.selection_border)
            vp_fill.setAlpha(45)
            painter.fillRect(QRectF(vx0, 1, vw, OVERVIEW_HEIGHT - 2), vp_fill)
            painter.setPen(QPen(QColor(palette.selection_border), 1.5))
            painter.drawRect(QRectF(vx0 + 0.5, 1.5, max(5.0, vw - 1.0), OVERVIEW_HEIGHT - 3))

        painter.setPen(QPen(QColor(palette.border), 1))
        painter.drawLine(0, OVERVIEW_HEIGHT - 1, width, OVERVIEW_HEIGHT - 1)

    def _paint_ruler(self, painter: QPainter) -> None:
        palette = theme.active()
        rect = QRect(0, OVERVIEW_HEIGHT, self.viewport().width(), RULER_HEIGHT)
        painter.fillRect(rect, QColor(palette.ruler))
        painter.setPen(QPen(QColor(palette.border), 1))
        painter.drawLine(0, HEADER_HEIGHT - 1, self.viewport().width(), HEADER_HEIGHT - 1)

        start, end = self._visible_range()
        interval = self._tick_interval()
        first = int(start / self._sample_rate / interval) * int(interval * self._sample_rate)
        font = QFont(painter.font().family(), 8)
        painter.setFont(font)
        tick = int(interval * self._sample_rate)
        if tick <= 0:
            return
        position = first
        while position <= end:
            x = self._frame_to_x(position)
            if x < -40:
                position += tick
                continue
            painter.setPen(QPen(QColor(palette.border), 1))
            painter.drawLine(QPointF(x, HEADER_HEIGHT - 6), QPointF(x, HEADER_HEIGHT))
            painter.setPen(QColor(palette.ruler_text))
            painter.drawText(QPointF(x + 3, HEADER_HEIGHT - 9),
                             frames_to_timestamp(position, self._sample_rate))
            position += tick

    def _paint_drag_overlay(self, painter: QPainter) -> None:
        """Show live timestamp and adjacent durations while dragging a divider."""
        if self._drag_origin is None or self._drag_frame is None:
            return
        palette = theme.active()
        rate = float(self._sample_rate or 44100)
        left, right = self._adjacent_tracks_for_boundary(self._drag_origin)
        ts = frames_to_timestamp(self._drag_frame, self._sample_rate)
        if left is not None and right is not None:
            l_dur = _format_duration((self._drag_frame - left.start_frame) / rate)
            r_dur = _format_duration((right.end_frame - self._drag_frame) / rate)
            label = f" {ts}   ({l_dur} | {r_dur}) "
        else:
            label = f" {ts} "
        font = QFont(painter.font())
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        tw = metrics.horizontalAdvance(label) + 8
        th = metrics.height() + 6
        x = self._frame_to_x(self._drag_frame)
        box_x = max(4.0, min(float(self.viewport().width()) - tw - 4.0, x - tw / 2.0))
        box_y = float(HEADER_HEIGHT + 24)
        box = QRectF(box_x, box_y, tw, th)
        painter.fillRect(box, QColor(palette.boundary_selected))
        painter.setPen(QColor(palette.background))
        painter.drawText(box, Qt.AlignCenter, label)

    def _tick_interval(self) -> float:
        """Choose a tick spacing of at least 80 pixels."""
        target = 80 * self._frames_per_pixel / max(1, self._sample_rate)
        for candidate in (0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600):
            if candidate >= target:
                return float(candidate)
        return 3600.0

    def _paint_cursor(self, painter: QPainter) -> None:
        """Draw the cursor and the playhead so they cannot be confused.

        The cursor is a thin dotted line: it is where the mouse is. The
        playhead is a solid bright line with a marker: it is where the
        sound is.
        """
        palette = theme.active()
        lanes = self._lane_rects()
        top = lanes[0].top()
        bottom = lanes[-1].top() + lanes[-1].height()

        x = self._frame_to_x(self._cursor)
        if 0 <= x <= self.viewport().width():
            painter.setPen(QPen(QColor(palette.text_faint), 1, Qt.DotLine))
            painter.drawLine(QPointF(x, top), QPointF(x, bottom))

        playhead_x = self._frame_to_x(self._playhead)
        if 0 <= playhead_x <= self.viewport().width():
            painter.setPen(QPen(QColor(palette.playhead), 1))
            painter.drawLine(QPointF(playhead_x, top), QPointF(playhead_x, bottom))
            painter.setBrush(QBrush(QColor(palette.playhead)))
            painter.setPen(Qt.NoPen)
            marker = QPainterPath()
            marker.moveTo(playhead_x - 5, top)
            marker.lineTo(playhead_x + 5, top)
            marker.lineTo(playhead_x, top + 8)
            marker.closeSubpath()
            painter.drawPath(marker)

    # ------------------------------------------------------------------
    # Peak access
    # ------------------------------------------------------------------
    def _span_peaks(self, start: int, end: int, buckets: int):
        channels = max(1, self._channels)
        mins = np.zeros((buckets, channels), dtype=np.float32)
        maxs = np.zeros((buckets, channels), dtype=np.float32)
        if end <= start:
            return mins, maxs
        span = float(end - start)

        cursor = 0
        for index, source in enumerate(self._sources):
            if index > 0:
                cursor += source.gap_frames
            seg_start = cursor
            seg_end = seg_start + source.frames
            cursor = seg_end

            a = max(start, seg_start)
            b = min(end, seg_end)
            if b <= a:
                continue
            pyramid = self._pyramids.get(source.source_id)
            if pyramid is None:
                continue
            i0 = int((a - start) / span * buckets)
            i1 = int((b - start) / span * buckets) + 1
            i0 = max(0, min(i0, buckets))
            i1 = max(i0 + 1, min(i1, buckets))
            local_mins, local_maxs = pyramid.peaks_for_span(a - seg_start, b - seg_start, i1 - i0)
            count = min(local_mins.shape[0], i1 - i0)
            mins[i0: i0 + count] = local_mins[:count]
            maxs[i0: i0 + count] = local_maxs[:count]
        return mins, maxs

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------
    def _pan_overview_to(self, x: float) -> None:
        if self._total_frames <= 0:
            return
        width = max(1, self.viewport().width())
        frac = max(0.0, min(1.0, x / float(width)))
        centre_frame = int(round(frac * self._total_frames))
        centre_pixel = int(centre_frame / self._frames_per_pixel)
        self.horizontalScrollBar().setValue(max(0, centre_pixel - width // 2))
        self.viewport().update()

    def _nearest_boundary(self, x: float) -> Optional[int]:
        tolerance = 8.0
        best = None
        best_distance = tolerance
        for track in self._tracks:
            for frame in (track.start_frame, track.end_frame):
                if frame <= 0 or frame >= self._total_frames:
                    continue
                distance = abs(self._frame_to_x(frame) - x)
                if distance <= best_distance:
                    best_distance = distance
                    best = frame
        return best

    def mousePressEvent(self, event) -> None:
        self.setFocus(Qt.MouseFocusReason)
        position = event.position()
        x = position.x()
        y = position.y()
        if event.button() == Qt.LeftButton:
            if y < OVERVIEW_HEIGHT and self._total_frames > 0:
                self._dragging_overview = True
                self._pan_overview_to(x)
                return
            if OVERVIEW_HEIGHT <= y < HEADER_HEIGHT and self._total_frames > 0:
                frame = max(0, min(self._x_to_frame(x), self._total_frames))
                self._cursor = frame
                self.cursorMoved.emit(frame)
                self.viewport().update()
                return
            boundary = self._nearest_boundary(x)
            if boundary is not None:
                left, right = self._adjacent_tracks_for_boundary(boundary)
                self._selected_boundary = boundary
                self.boundaryClicked.emit(boundary)
                if left is not None and right is not None:
                    if left.locked or right.locked:
                        QToolTip.showText(
                            event.globalPosition().toPoint(),
                            "Locked — unlock adjacent tracks to move this divider",
                            self,
                        )
                    else:
                        min_span = min(max(1, int(0.05 * (self._sample_rate or 44100))),
                                       max(1, (right.end_frame - left.start_frame) // 4))
                        self._drag_origin = boundary
                        self._drag_frame = boundary
                        self._drag_bounds = (left.start_frame + min_span, right.end_frame - min_span)
                        self.viewport().setCursor(Qt.SplitHCursor)
                self.viewport().update()
                return
            frame = max(0, min(self._x_to_frame(x), self._total_frames))
            self._cursor = frame
            track = self._track_at(frame)
            if track is not None:
                self._selected_track_id = track.track_id
                self.trackSelected.emit(track.track_id)
            self.viewport().update()

    def mouseMoveEvent(self, event) -> None:
        position = event.position()
        x = position.x()
        y = position.y()
        if self._dragging_overview:
            self._pan_overview_to(x)
            return
        if self._drag_origin is not None:
            # Auto-pan smoothly when dragging a divider near/past viewport edges while zoomed in.
            width = max(1, self.viewport().width())
            bar = self.horizontalScrollBar()
            edge_margin = 20.0
            if bar.maximum() > 0:
                if x < edge_margin:
                    step = max(4, int((edge_margin - x) * 0.6))
                    bar.setValue(max(bar.minimum(), bar.value() - step))
                elif x > width - edge_margin:
                    step = max(4, int((x - (width - edge_margin)) * 0.6))
                    bar.setValue(min(bar.maximum(), bar.value() + step))
            frame = self._x_to_frame(x)
            if self._drag_bounds is not None:
                lo, hi = self._drag_bounds
                frame = max(lo, min(frame, hi))
            else:
                frame = max(0, min(frame, self._total_frames))
            self._drag_frame = frame
            self.viewport().setCursor(Qt.SplitHCursor)
            self.viewport().update()
            return
        if y < OVERVIEW_HEIGHT:
            self.viewport().setCursor(Qt.PointingHandCursor)
            if self._hover_boundary is not None:
                self._hover_boundary = None
                QToolTip.hideText()
                self.viewport().update()
            return
        boundary = self._nearest_boundary(x) if y >= HEADER_HEIGHT else None
        if boundary is not None:
            left, right = self._adjacent_tracks_for_boundary(boundary)
            locked = bool((left and left.locked) or (right and right.locked))
            self.viewport().setCursor(Qt.ForbiddenCursor if locked else Qt.SplitHCursor)
        else:
            self.viewport().unsetCursor()
        if boundary != self._hover_boundary:
            self._hover_boundary = boundary
            if boundary is not None:
                left, right = self._adjacent_tracks_for_boundary(boundary)
                locked = bool((left and left.locked) or (right and right.locked))
                hint = "Locked — unlock adjacent tracks to drag" if locked else "Drag to move shared divider"
                reasons = self._reasons.get(boundary, [])
                extra = ("\n" + "\n".join(reasons)) if reasons else ""
                QToolTip.showText(event.globalPosition().toPoint(),
                                  f"Divider at {frames_to_timestamp(boundary, self._sample_rate)}\n{hint}{extra}",
                                  self)
            else:
                QToolTip.hideText()
            self.viewport().update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        if self._dragging_overview:
            self._dragging_overview = False
            return
        if self._drag_origin is not None and self._drag_frame is not None:
            origin = self._drag_origin
            target = self._drag_frame
            self._drag_origin = None
            self._drag_frame = None
            self._drag_bounds = None
            self.viewport().unsetCursor()
            if target != origin:
                self.boundaryDragged.emit(origin, target)
            self.viewport().update()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or self._total_frames <= 0:
            return
        x = event.position().x()
        y = event.position().y()
        if y < OVERVIEW_HEIGHT:
            return
        boundary = self._nearest_boundary(x)
        if boundary is not None:
            self._selected_boundary = boundary
            self.boundaryClicked.emit(boundary)
            return
        frame = max(0, min(self._x_to_frame(x), self._total_frames))
        self._cursor = frame
        track = self._track_at(frame)
        if track is not None:
            self._selected_track_id = track.track_id
            self.trackSelected.emit(track.track_id)
        self.seekRequested.emit(frame)
        self.viewport().update()

    def contextMenuEvent(self, event) -> None:
        if self._total_frames <= 0:
            return
        pos = event.pos()
        x = float(pos.x())
        frame = max(0, min(self._x_to_frame(x), self._total_frames))
        boundary = self._nearest_boundary(x)
        track = self._track_at(frame)
        menu = QMenu(self)
        if boundary is not None:
            remove_act = menu.addAction("Remove split (merge adjacent tracks)")
            remove_act.triggered.connect(lambda: self.removeSplitRequested.emit(boundary))
            menu.addSeparator()
        if track is not None and track.start_frame < frame < track.end_frame:
            split_act = menu.addAction(
                f"Split here ({frames_to_timestamp(frame, self._sample_rate)})"
            )
            split_act.triggered.connect(lambda: self.splitRequested.emit(frame))
        if track is not None:
            rename_act = menu.addAction("Rename track…")
            rename_act.triggered.connect(lambda: self.renameTrackRequested.emit(track.track_id))
            play_act = menu.addAction("Play this track")
            play_act.triggered.connect(lambda: self.playTrackRequested.emit(track.track_id))
            zoom_act = menu.addAction("Zoom to this track")
            zoom_act.triggered.connect(lambda: self.zoomTrackRequested.emit(track.track_id))
            menu.addSeparator()
            lock_label = "Unlock track" if track.locked else "Lock track"
            lock_act = menu.addAction(lock_label)
            lock_act.triggered.connect(lambda: self.toggleLockRequested.emit(track.track_id))
        if not menu.isEmpty():
            menu.exec(event.globalPos())

    def wheelEvent(self, event) -> None:
        modifiers = event.modifiers()
        delta = event.angleDelta().y()
        if delta == 0:
            return
        if modifiers & (Qt.ControlModifier | Qt.ShiftModifier):
            anchor = self._x_to_frame(event.position().x())
            self.zoom_by(0.85 if delta > 0 else 1.18, anchor)
        else:
            bar = self.horizontalScrollBar()
            bar.setValue(bar.value() + (3 if delta > 0 else -3) * bar.singleStep())
        event.accept()

    def cancel_drag(self) -> bool:
        """Cancel an in-progress divider or overview drag. Returns True if one was active."""
        if self._drag_origin is None and not self._dragging_overview:
            return False
        self._drag_origin = None
        self._drag_frame = None
        self._drag_bounds = None
        self._dragging_overview = False
        self.viewport().unsetCursor()
        self.viewport().update()
        return True

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Escape and self.cancel_drag():
            event.accept()
            return
        if key in (Qt.Key_Left, Qt.Key_Right):
            step = int(10 * self._sample_rate / 1000.0) * self._frames_per_pixel
            self._cursor = max(0, min(self._cursor + (step if key == Qt.Key_Right else -step),
                                      self._total_frames))
            self.cursorMoved.emit(self._cursor)
            self.viewport().update()
            event.accept()
            return
        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_by(0.8, self._cursor)
            event.accept()
            return
        if key == Qt.Key_Minus:
            self.zoom_by(1.25, self._cursor)
            event.accept()
            return
        if key == Qt.Key_0:
            self.zoom_to_fit()
            event.accept()
            return

    def _track_at(self, frame: int) -> Optional[Track]:
        for track in self._tracks:
            if track.start_frame <= frame < track.end_frame:
                return track
        return None
