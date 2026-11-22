import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from splytnyt.ui.main_window import MainWindow
from splytnyt.core.model import Track

@pytest.fixture
def window():
    app=QApplication.instance() or QApplication([])
    w=MainWindow(); w._autosave_timer.stop() if hasattr(w,'_autosave_timer') else None
    w.session.project.tracks=[Track(start_frame=i*480000,end_frame=(i+1)*480000,title=f'Track {i+1}') for i in range(4)]
    w._reload();w.show();app.processEvents()
    yield w
    w.session._dirty=False;w.close();app.processEvents()

def test_click_keyboard_and_waveform_share_selection_without_seeking(window):
    w=window; w.waveform.set_cursor(12345)
    def selected(row):
        track=w.session.project.tracks[row]
        assert w.table.selected_track().track_id==w._selected_track_id==w.waveform._selected_track_id==track.track_id
        assert w.title_edit.text()==track.title
        assert w.waveform.cursor_frame==12345
    index=w.table.model().index(2,0)
    QTest.mouseClick(w.table.viewport(),Qt.LeftButton,pos=w.table.visualRect(index).center());selected(2)
    # Reproduces the old bug: selecting via waveform left currentIndex on row 2.
    w._on_track_selected(w.session.project.tracks[0].track_id);selected(0)
    QTest.keyClick(w.table,Qt.Key_Down);selected(1)
    played=[]
    w._play_range=lambda start,end,message:played.append((start,end))
    w.play_track_button.click()
    assert played==[(480000,960000)]
    w._reload();selected(1)

def test_double_click_seek_is_explicit_and_editable_title_does_not_seek(window):
    w=window; w.waveform.set_cursor(12345)
    w.table._on_double_click(w.table.model().index(2,1))
    assert w.waveform.cursor_frame==12345
    w.table._on_double_click(w.table.model().index(2,0))
    assert w.waveform.cursor_frame==960000
    assert w._selected_track_id==w.session.project.tracks[2].track_id

def test_ordinary_play_uses_cursor_and_selected_play_uses_selected_range(window):
    from types import SimpleNamespace
    w=window; played=[]
    w._playback=SimpleNamespace(is_playing=False,play=lambda *a:played.append(a),stop=lambda:None)
    w._on_track_selected(w.session.project.tracks[3].track_id)
    w.waveform.set_cursor(12345);w.toggle_playback()
    assert played[-1]==(12345,None)
    w.play_selected_track();assert played[-1]==(1440000,1920000)
    w._playback=None
