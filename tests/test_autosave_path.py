from types import SimpleNamespace
import pytest
from splytnyt.app.session import Session
from splytnyt.ui.main_window import MainWindow
from splytnyt.core.model import Source, Track
from splytnyt.core.serialize import load_project

@pytest.mark.parametrize('named',[False,True])
def test_autosave_never_redirects_the_user_save_path(tmp_path,named):
    session=Session()
    session.project.sources=[Source(path='read-only-source.wav',sample_rate=48000,channels=2,frames=48000)]
    session.project.tracks=[Track(start_frame=0,end_frame=48000)]
    original=str(tmp_path/'review.splytnyt') if named else ''
    session.project.path=original
    session.mark_dirty()
    recovery=tmp_path/'recovery.splytnyt'
    window=SimpleNamespace(session=session,_recovery_path=lambda:str(recovery),_last_autosave=0)
    MainWindow._autosave(window)
    assert recovery.exists()
    assert session.project.path==original and session.dirty
    assert len(load_project(str(recovery)).tracks)==1


def test_recovery_prompt_waits_until_main_window_is_visible(tmp_path,monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtWidgets import QApplication
    app=QApplication.instance() or QApplication([])
    calls=[]
    monkeypatch.setattr(MainWindow,'_offer_recovery',lambda self:calls.append(self.isVisible()))
    monkeypatch.setattr(MainWindow,'_recovery_path',lambda self:str(tmp_path/'recovery.splytnyt'))
    window=MainWindow()
    assert calls==[]
    window.show();app.processEvents()
    assert calls==[True]
    window.close()
