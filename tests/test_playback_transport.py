"""Exercise callback transport with deterministic device and decoder failures.
These tests prove routing/lifecycle, not physical audibility.
"""
import threading,time
import numpy as np
import pytest
import splytnyt.audio.playback as playback

# The transport is driven through a fake device, but the module still needs
# an audio backend to be importable: without PortAudio there is no
# ``playback.sd`` to replace, and there is nothing to test.
pytestmark = pytest.mark.skipif(
    playback.sd is None,
    reason="no audio backend on this machine (PortAudio is not available)",
)

class Reader:
    sample_rate=48000
    channels=2
    total_frames=9000
    def read(self,start,count):
        return np.repeat(np.arange(start,start+count,dtype=np.float32)[:,None],2,axis=1)

class Device:
    def __init__(self,callback,finished_callback,**kwargs):
        self.callback=callback;self.finished=finished_callback;self.blocks=[]
        self.halt=threading.Event();self.thread=None
    def start(self):
        def pump():
            try:
                while not self.halt.is_set():
                    block=np.zeros((480,2),dtype=np.float32)
                    try:self.callback(block,480,None,None)
                    finally:self.blocks.append(block.copy())
                    time.sleep(.001)
            except (playback.sd.CallbackStop,playback.sd.CallbackAbort):pass
            finally:self.finished()
        self.thread=threading.Thread(target=pump);self.thread.start()
    def abort(self):
        self.halt.set()
        if self.thread:self.thread.join(1)
    def close(self):pass


def test_selected_range_reaches_device_in_order_and_stream_closes(monkeypatch):
    devices=[]
    def make(**kwargs):
        d=Device(**kwargs);devices.append(d);return d
    monkeypatch.setattr(playback.sd,'OutputStream',make)
    monkeypatch.setattr(playback.sd,'query_devices',lambda *a:{'name':'test'})
    monkeypatch.setattr(playback.sd,'check_output_settings',lambda **k:None)
    engine=playback.PlaybackEngine(Reader(),device=8)
    engine.play(700,8000)
    deadline=time.monotonic()+3
    while engine.is_playing and time.monotonic()<deadline:time.sleep(.01)
    assert not engine.is_playing and not engine.error
    assert engine.position==8000
    assert engine.diagnostics['decoded_frames']==7300
    assert engine.diagnostics['queued_frames']==7300
    assert engine.diagnostics['callback_peak']>0
    sent=np.concatenate(devices[0].blocks)[:7300,0]
    np.testing.assert_array_equal(sent,np.arange(700,8000,dtype=np.float32))
    engine.close()
    assert not devices[0].thread.is_alive()


def test_decoder_failure_reaches_error_callback(monkeypatch):
    class Broken(Reader):
        def read(self,*a):raise RuntimeError('damaged source')
    monkeypatch.setattr(playback.sd,'query_devices',lambda *a:{'name':'test endpoint'})
    monkeypatch.setattr(playback.sd,'check_output_settings',lambda **k:None)
    errors=[];engine=playback.PlaybackEngine(Broken(),device=8,on_error=errors.append)
    engine.play(0)
    deadline=time.monotonic()+2
    while engine.is_playing and time.monotonic()<deadline:time.sleep(.01)
    assert errors and 'damaged source' in errors[0] and 'test endpoint' in errors[0]
    assert not engine.is_playing


def test_callback_error_is_reported_even_if_finished_fires_immediately(monkeypatch):
    class FailedDevice(Device):
        def start(self):
            # A malformed endpoint callback fails before the worker can poll it.
            try:self.callback(np.zeros((480,3),dtype=np.float32),480,None,None)
            except playback.sd.CallbackAbort:pass
            self.finished()
    monkeypatch.setattr(playback.sd,'OutputStream',lambda **kw:FailedDevice(**kw))
    monkeypatch.setattr(playback.sd,'query_devices',lambda *a:{'name':'test endpoint'})
    monkeypatch.setattr(playback.sd,'check_output_settings',lambda **k:None)
    errors=[];engine=playback.PlaybackEngine(Reader(),device=8,on_error=errors.append)
    engine.play(0)
    deadline=time.monotonic()+2
    while engine.is_playing and time.monotonic()<deadline:time.sleep(.01)
    assert errors and 'broadcast' in errors[0]
    assert engine.diagnostics['callback_error']
    engine.close()


def test_stale_worker_stop_does_not_stop_replacement_playback_ui():
    from types import SimpleNamespace
    from splytnyt.ui.main_window import MainWindow
    class Timer:
        running=False
        def start(self):self.running=True
        def stop(self):self.running=False
    class Button:
        def setText(self,text):self.text=text
        def setIcon(self,icon):pass
    timer=Timer();button=Button()
    window=SimpleNamespace(_playback=SimpleNamespace(is_playing=True),
                           _playback_timer=timer,play_button=button,
                           style=lambda:SimpleNamespace(standardIcon=lambda value:None))
    MainWindow._on_playback_state(window,False)
    assert timer.running and button.text=='Pause'
    window._playback.is_playing=False
    MainWindow._on_playback_state(window,True)
    assert not timer.running and button.text=='Play'
