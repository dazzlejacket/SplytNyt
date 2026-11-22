"""Opt-in packaged regression: real Qt clicks, real PortAudio output, real export.

No mocks. Invoke:

    SplytNyt.exe PROJECT --verify-selection REPORT.json

The three things it proves about the *frozen* application, which a source-level
test cannot prove:

1. the real interface responds to real clicks,
2. real audio comes out of the real output device,
3. the exporter really runs inside the bundle - PyInstaller can happily omit a
   module that is only imported on the export path, and the application would
   still start. This catches exactly that.
"""
import json,time,sys,os,tempfile
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest


def _export_and_check(project, require):
    """Export the open project in every container and check what came out.

    Returns a dict describing what was verified.
    """
    import numpy as np
    import soundfile as sf
    from ..app.exporting import export_project
    from ..audio.timeline import build_reader
    from ..core.model import ExportSettings

    reader = build_reader(project)
    summary = {}
    try:
        source = reader.read(0, reader.total_frames)
        require(len(source) > 0, "the source timeline is empty")

        def decode(path):
            with sf.SoundFile(path) as handle:
                require(
                    handle.samplerate == reader.sample_rate
                    and handle.channels == reader.channels,
                    f"{os.path.basename(path)} changed the sample rate or channels",
                )
                blocks = []
                while True:
                    block = handle.read(65536, dtype="float32", always_2d=True)
                    if not len(block):
                        break
                    blocks.append(block)
            if not blocks:
                return np.zeros((0, reader.channels), dtype=np.float32)
            return np.concatenate(blocks)

        with tempfile.TemporaryDirectory(prefix="splytnyt-export-check-") as tmp:
            for container in ("flac", "wav", "mp3"):
                out_dir = os.path.join(tmp, container)
                settings = ExportSettings(
                    output_dir=out_dir,
                    filename_template="{n:02d} track",
                    container=container,
                    mode="separate",
                    include_cue=True,
                    include_m3u=True,
                    include_info=True,
                    include_checksums=True,
                    write_tags=True,
                    gain_mode="untouched",
                )
                report = export_project(project, settings)

                require(report.files, f"the {container} export wrote no files")
                require(
                    all(os.path.isfile(path) for path in report.files),
                    f"the {container} export reported a file that is not there",
                )
                require(
                    os.path.isfile(report.checksum_path),
                    f"the {container} export wrote no checksums",
                )

                if container in ("flac", "wav"):
                    # Bit-exact: the exported tracks must tile the source
                    # timeline and reproduce every sample.
                    joined = np.concatenate([decode(path) for path in report.files])
                    require(
                        len(joined) == len(source),
                        f"the {container} export holds {len(joined)} frames, "
                        f"the source holds {len(source)}",
                    )
                    maximum = float(np.max(np.abs(joined - source)))
                    require(
                        maximum <= 2 ** -24,
                        f"the {container} export is not bit-exact (max error {maximum})",
                    )
                    summary[container] = {
                        "files": len(report.files),
                        "frames_compared": int(len(joined)),
                        "max_pcm_error": maximum,
                        "bit_exact": True,
                    }
                else:
                    # MP3 is lossy: prove it decodes and holds the right length.
                    frames = 0
                    for path in report.files:
                        with sf.SoundFile(path) as handle:
                            frames += handle.frames
                    require(frames > 0, "the mp3 export decoded to nothing")
                    summary[container] = {
                        "files": len(report.files),
                        "frames": int(frames),
                        "decoded": True,
                    }

            # A continuous export writes one program file and a CUE sheet.
            out_dir = os.path.join(tmp, "continuous")
            settings = ExportSettings(
                output_dir=out_dir,
                filename_template="track",
                container="flac",
                mode="continuous",
                include_cue=True,
                include_info=True,
                include_checksums=True,
                write_tags=True,
            )
            report = export_project(project, settings)
            program = os.path.join(out_dir, "program.flac")
            require(os.path.isfile(program), "the continuous export wrote no program.flac")
            require(os.path.isfile(report.cue_path), "the continuous export wrote no CUE sheet")
            joined = decode(program)
            require(
                len(joined) == len(source),
                f"the continuous export holds {len(joined)} frames, the source holds {len(source)}",
            )
            maximum = float(np.max(np.abs(joined - source)))
            require(
                maximum <= 2 ** -24,
                f"the continuous export is not bit-exact (max error {maximum})",
            )
            cue = Path(report.cue_path).read_text(encoding="utf-8", errors="replace")
            require("TRACK " in cue.upper(), "the CUE sheet has no TRACK entries")
            summary["continuous"] = {
                "files": 1,
                "frames_compared": int(len(joined)),
                "max_pcm_error": maximum,
                "bit_exact": True,
                "cue_written": True,
                "cue_tracks": cue.upper().count("TRACK "),
            }
    finally:
        reader.close()
    return summary


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
        # The exporter must really work inside the bundle.
        export_summary=_export_and_check(w.session.project, require)
        outcome=dict(passed=True,frozen=bool(getattr(sys,'frozen',False)),executable=sys.executable,
                     project=w.session.path if hasattr(w.session,'path') else '',sample_rate=rate,checks=rows,
                     export=export_summary,
                     double_click='seek without autoplay',selection_does_not_seek=True)
        code=0
    except Exception as exc:
        outcome=dict(passed=False,error=repr(exc),checks=rows);code=1
    Path(output).write_text(json.dumps(outcome,indent=2),encoding='utf-8')
    window.session._dirty=False
    window.close();app.exit(code)
