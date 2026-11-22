"""Playback.

Uses PortAudio through :mod:`sounddevice`. Playback pulls samples from the
same :class:`~splytnyt.audio.timeline.TimelineReader` that the exporter uses,
so what you hear is what gets written.

This module is deliberately free of Qt: it reports progress through plain
callbacks so it can be driven from a worker thread or from a test.
"""

from __future__ import annotations

import threading
import time
import queue
import sys
import json
import os
import tempfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:  # pragma: no cover - depends on the host having PortAudio
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
    SOUNDDEVICE_ERROR = ""
except Exception as exc:  # pragma: no cover
    sd = None
    SOUNDDEVICE_AVAILABLE = False
    SOUNDDEVICE_ERROR = str(exc)

from .timeline import TimelineReader

#: Frames per write chunk during playback (~21 ms at 48 kHz).
CHUNK_FRAMES = 1024

PositionCallback = Optional[Callable[[int], None]]
StateCallback = Optional[Callable[[bool], None]]


class PlaybackError(RuntimeError):
    """Raised when audio output is unavailable or a playback cannot start."""


class PlaybackEngine:
    """Streams the timeline to the default output device."""

    def __init__(
        self,
        reader: TimelineReader,
        on_position: PositionCallback = None,
        on_state: StateCallback = None,
        on_error: Optional[Callable[[str], None]] = None,
        device: Optional[int] = None,
    ) -> None:
        self._reader = reader
        self._on_position = on_position
        self._on_state = on_state
        self._on_error = on_error
        self.device = device
        self.device_name = ""

        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._position = 0
        self._playing = False
        self._start_frame = 0
        self._end_frame: Optional[int] = None
        self._error = ""
        self._stream = None
        self.diagnostics = {}
        self.trace_path = Path(os.environ.get('SPLYTNYT_PLAYBACK_TRACE',
                                             str(Path(tempfile.gettempdir()) / 'SplytNyt-playback.jsonl')))

    def _trace(self, event):
        """Write outside the audio callback; a stream open is not output proof."""
        snapshot = dict(self.diagnostics, event=event, position=self.position,
                        error=self._error, timestamp=time.time())
        try:
            with self.trace_path.open('a', encoding='utf-8') as log:
                log.write(json.dumps(snapshot) + '\n')
        except OSError:
            pass  # inability to write a diagnostic must not stop playback

    # ------------------------------------------------------------------
    @property
    def sample_rate(self) -> int:
        return self._reader.sample_rate

    @property
    def channels(self) -> int:
        return self._reader.channels

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    @property
    def position(self) -> int:
        """Current playhead in global timeline frames."""
        with self._lock:
            return self._position

    @property
    def error(self) -> str:
        return self._error

    # ------------------------------------------------------------------
    def play(self, start_frame: int, end_frame: Optional[int] = None) -> None:
        """Start playing from ``start_frame`` until ``end_frame`` (or the end)."""
        if not SOUNDDEVICE_AVAILABLE:
            raise PlaybackError(
                "No audio output device is available: " + SOUNDDEVICE_ERROR
            )
        self.stop()

        total = self._reader.total_frames
        start_frame = max(0, min(int(start_frame), max(0, total - 1)))
        if end_frame is not None:
            end_frame = int(end_frame)
            if end_frame <= start_frame:
                return
            end_frame = min(end_frame, total)
        else:
            end_frame = total

        with self._lock:
            self._start_frame = start_frame
            self._end_frame = end_frame
            self._position = start_frame
            self._playing = True
            self._error = ""

        self._stop_event.clear()
        self.diagnostics = dict(start_frame=start_frame, end_frame=end_frame,
                                sample_rate=self.sample_rate, channels=self.channels,
                                decoded_frames=0, decoded_peak=0.0, queued_frames=0,
                                callbacks=0, callback_frames=0, callback_peak=0.0,
                                underflows=0, queue_starvations=0, callback_error='')
        self._thread = threading.Thread(target=self._run, name="splytnyt-playback", daemon=True)
        self._notify_state(True)
        self._thread.start()

    def stop(self) -> None:
        """Stop playback and wait briefly for the worker to stop.

        A device that is busy can leave the worker blocked inside a write,
        so the stream is aborted as well: that is the way PortAudio is told
        to let go of a write that is in progress. Without it the worker can
        outlive the stop, and two workers writing at once brings the
        application down.
        """
        self._stop_event.set()
        stream = self._stream
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        if thread is not None and thread.is_alive():
            raise PlaybackError("The previous audio stream has not stopped. Reconnect the output device and restart SplytNyt.")
        self._thread = None
        with self._lock:
            was_playing = self._playing
            self._playing = False
        if was_playing:
            self._notify_state(False)

    def set_position(self, frame: int) -> None:
        """Move the playhead (only meaningful while stopped)."""
        with self._lock:
            if self._playing:
                return
            self._position = max(0, int(frame))

    def close(self) -> None:
        self.stop()

    # ------------------------------------------------------------------
    def _run(self) -> None:
        stream = None
        com_initialized = False
        if sys.platform == "win32":
            # WASAPI uses COM on the thread that opens/starts the stream.
            import ctypes
            com_initialized = ctypes.windll.ole32.CoInitializeEx(None, 0) in (0, 1)
        buffers = queue.Queue(maxsize=32)
        finished = threading.Event()
        last_callback = [time.monotonic()]
        callback_error = []
        pending = [None, 0]
        submitted = [self._start_frame]

        def callback(outdata, frames, timing, status):
            outdata.fill(0)
            last_callback[0] = time.monotonic()
            stats = self.diagnostics
            stats['callbacks'] += 1
            stats['callback_frames'] += frames
            if status and status.output_underflow:
                stats['underflows'] += 1
            if self._stop_event.is_set():
                raise sd.CallbackAbort
            offset = 0
            try:
                while offset < frames:
                    if pending[0] is None:
                        try:
                            pending[0] = buffers.get_nowait()
                            pending[1] = 0
                        except queue.Empty:
                            stats['queue_starvations'] += 1
                            return  # producer catches up; no decoder I/O on audio thread
                        if pending[0] is False:
                            raise sd.CallbackStop
                    block = pending[0]
                    take = min(frames - offset, len(block) - pending[1])
                    outdata[offset:offset + take] = block[pending[1]:pending[1] + take]
                    stats['callback_peak'] = max(stats['callback_peak'],
                                                float(np.max(np.abs(outdata[offset:offset + take]))))
                    pending[1] += take
                    offset += take
                    submitted[0] += take
                    if pending[1] == len(block):
                        pending[0] = None
                with self._lock:
                    self._position = submitted[0]
            except (sd.CallbackStop, sd.CallbackAbort):
                with self._lock:
                    self._position = submitted[0]
                raise
            except Exception as exc:
                callback_error.append(str(exc))
                stats['callback_error'] = repr(exc)
                raise sd.CallbackAbort

        def enqueue(cursor, count):
            block = self._reader.read(cursor, count)
            if len(block) != count:
                raise PlaybackError(f'Decoder returned {len(block)} of {count} frames at {cursor}.')
            if block.shape != (count, self.channels) or not np.isfinite(block).all():
                raise PlaybackError(f'Invalid decoded audio buffer at frame {cursor}: {block.shape}.')
            self.diagnostics['decoded_frames'] += count
            self.diagnostics['decoded_peak'] = max(self.diagnostics['decoded_peak'], float(np.max(np.abs(block))))
            buffers.put_nowait(block)
            self.diagnostics['queued_frames'] += count

        try:
            device = self.device
            if device is None:
                # The WASAPI default follows Windows' chosen endpoint, including
                # headphones. Legacy MME can block forever in Pa_WriteStream.
                for api in sd.query_hostapis():
                    if api['name'] == 'Windows WASAPI' and api['default_output_device'] >= 0:
                        device = api['default_output_device']
                        break
                if sys.platform == 'win32' and device is None:
                    raise PlaybackError('Windows has no active WASAPI default output. '
                                        'Select an available endpoint in Playback > Output device.')
            info = sd.query_devices(device, 'output')
            self.device_name = info['name']
            self.diagnostics.update(device=device, device_name=self.device_name,
                                    hostapi=sd.query_hostapis(info['hostapi'])['name'] if 'hostapi' in info else '',
                                    device_channels=info.get('max_output_channels'),
                                    device_default_rate=info.get('default_samplerate'))
            output_options = {}
            if 'hostapi' in info and sd.query_hostapis(info['hostapi'])['name'] == 'Windows WASAPI':
                # Shared-mode Windows conversion is for monitoring only. The
                # timeline and exported sample rate remain source-native.
                output_options['extra_settings'] = sd.WasapiSettings(auto_convert=True)
            sd.check_output_settings(device=device, samplerate=self.sample_rate,
                                     channels=self.channels, dtype='float32', **output_options)
            self.diagnostics['format_supported'] = True
            cursor = self._start_frame
            end = self._end_frame
            # Prime before starting, then decode ahead in this worker only.
            while not buffers.full() and cursor < end:
                count = min(CHUNK_FRAMES, end - cursor)
                enqueue(cursor, count)
                cursor += count
            stream = sd.OutputStream(device=device, samplerate=self.sample_rate,
                                     channels=self.channels, dtype='float32',
                                     blocksize=0, latency="high", callback=callback,
                                     finished_callback=finished.set, **output_options)
            with self._lock:
                self._stream = stream
            stream.start()
            self._trace('started')
            last_trace = time.monotonic()
            ended = False
            last_position = self._position
            last_advance = time.monotonic()
            while not self._stop_event.is_set() and not finished.is_set():
                if callback_error:
                    raise PlaybackError(callback_error[0])
                now = time.monotonic()
                if now - last_trace >= 0.5:
                    self.diagnostics.update(stream_active=bool(stream.active),
                                            stream_time=float(stream.time),
                                            callback_age=now-last_callback[0], queue_blocks=buffers.qsize())
                    self._trace('progress')
                    last_trace = now
                if self._position != last_position:
                    last_position = self._position
                    last_advance = now
                    if self._on_position:
                        self._on_position(last_position)
                if now - last_advance > 2.5:
                    raise PlaybackError('The output device stopped consuming audio. '
                                        'Choose another output in Playback > Output device, '
                                        'or reconnect the device and retry.')
                if not buffers.full() and not ended:
                    if cursor < end:
                        count = min(CHUNK_FRAMES, end - cursor)
                        enqueue(cursor, count)
                        cursor += count
                    else:
                        buffers.put_nowait(False)
                        ended = True
                else:
                    self._stop_event.wait(0.005)
            # finished_callback can fire before the worker re-enters its loop.
            # Preserve callback failures even when the stream has already ended.
            if callback_error and not self._stop_event.is_set():
                raise PlaybackError(callback_error[0])
        except Exception as exc:
            if not self._stop_event.is_set():
                with self._lock:
                    self._error = (f'Playback failed on {self.device_name or "the default output"}: {exc} '
                                   'Check the Windows volume and output connection, then choose '
                                   f'Playback > Output device and retry. Diagnostic log: {self.trace_path}')
                if self._on_error:
                    self._on_error(self._error)
        finally:
            if stream is not None:
                try:
                    stream.abort()
                    stream.close()
                except Exception:
                    pass
            self.diagnostics['stream_active'] = False
            self._trace('stopped')
            with self._lock:
                self._stream = None
                self._playing = False
            self._notify_state(False)
            if com_initialized:
                ctypes.windll.ole32.CoUninitialize()

    def _notify_state(self, playing: bool) -> None:
        if self._on_state:
            try:
                self._on_state(playing)
            except Exception:
                pass


def list_output_devices() -> list[dict]:
    """Available output devices, for the settings dialog."""
    if not SOUNDDEVICE_AVAILABLE:
        return []
    devices = []
    for index, device in enumerate(sd.query_devices()):
        if device.get("max_output_channels", 0) > 0:
            devices.append(
                {
                    "index": index,
                    "name": device.get("name", ""),
                    "channels": int(device.get("max_output_channels", 0)),
                    "default_samplerate": float(device.get("default_samplerate", 44100)),
                    "hostapi": sd.query_hostapis(device['hostapi'])['name'],
                }
            )
    return devices
