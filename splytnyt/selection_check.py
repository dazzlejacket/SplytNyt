"""Opt-in packaged regression: real Qt clicks and real PortAudio output.

No mocks. Invoke SplytNyt.exe PROJECT --verify-selection REPORT.json.
Use endpoint loopback alongside this check to independently identify output PCM.
"""
import json,time,sys
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest


def verify(window, output, app):
    rows=[]
    def require(condition, message):
        if not condition:raise RuntimeError(message)
    try:
        w=window;rate=w.session.project.sample_rate()
        require(w._playback is not None and len(w.session.project.tracks)>=4, 'Audio or tracks missing')
        w.waveform.set_cursor(60*rate)
        def select(row,keyboard=False):
            old=w.waveform.cursor_frame
            if keyboard:QTest.keyClick(w.table,Qt.Key_Down)
            else:
                index=w.table.model().index(row,0);w.table.scrollTo(index);app.processEvents()
                QTest.mouseClick(w.table.viewport(),Qt.LeftButton,pos=w.table.visualRect(index).center())
            app.processEvents();track=w.table.track_at_row(row)
            require(w._selected_track_id==w.table.selected_track().track_id==w.waveform._selected_track_id==track.track_id, 'Selection mismatch')
            require(w.title_edit.text()==track.title and w.waveform.cursor_frame==old, 'Details or cursor changed unexpectedly')
            return track
        def audio(label,track=None):
            started=time.time()
            if track:QTest.mouseClick(w.play_track_button,Qt.LeftButton)
            else:QTest.mouseClick(w.play_button,Qt.LeftButton)
            QTest.qWait(2600)
            require(w._playback.is_playing and not w._playback.error, 'Playback not running: '+w._playback.error)
            stats=dict(w._playback.diagnostics)
            require(stats['callbacks']>0 and stats['callback_peak']>0, 'No callback audio')
            if track:require(stats['start_frame']==track.start_frame and stats['end_frame']==track.end_frame, 'Wrong selected range')
            else:require(stats['start_frame']==60*rate, 'Ordinary Play did not retain cursor')
            w.stop_playback();QTest.qWait(200)
            rows.append(dict(case=label,timestamp=started,diagnostics=stats,endpoint=w._playback.device_name))
        track=select(2);require(not w._playback.is_playing, 'Selection autoplayed')
        audio('single click then Play selected',track)
        track=select(3,keyboard=True);audio('keyboard Down then Play selected',track)
        w.waveform.set_cursor(60*rate);select(1);audio('ordinary Play retains playhead')
        # Double-click is a stopped seek, not an implicit Play.
        index=w.table.model().index(2,0)
        QTest.mouseClick(w.table.viewport(),Qt.LeftButton,pos=w.table.visualRect(index).center())
        QTest.mouseDClick(w.table.viewport(),Qt.LeftButton,pos=w.table.visualRect(index).center())
        app.processEvents()
        require(w.waveform.cursor_frame==w.table.track_at_row(2).start_frame and not w._playback.is_playing, 'Double-click did not seek while stopped')
        outcome=dict(passed=True,frozen=bool(getattr(sys,'frozen',False)),executable=sys.executable,
                     project=w.session.path if hasattr(w.session,'path') else '',sample_rate=rate,checks=rows,
                     double_click='seek without autoplay',selection_does_not_seek=True)
        code=0
    except Exception as exc:
        outcome=dict(passed=False,error=repr(exc),checks=rows);code=1
    Path(output).write_text(json.dumps(outcome,indent=2),encoding='utf-8')
    window.session._dirty=False
    window.close();app.exit(code)
