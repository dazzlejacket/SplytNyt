"""Review lyric evidence without changing the edited timeline."""
from __future__ import annotations
import json
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QLabel,QTextEdit,
    QPushButton,QFileDialog,QMessageBox,QTreeWidget,QTreeWidgetItem)
from ..analysis.lyrics import parse_references


class LyricInputDialog(QDialog):
    def __init__(self, references, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Reference lyrics — local import'); self.resize(780,600)
        layout=QVBoxLayout(self)
        label=QLabel('Paste blocks beginning # Song title, then Source: URL or description, Version: production, and lyric lines.\n'
                     'Or import splytnyt-lyrics/1 JSON with full lyrics or short anchors. No online lyrics are downloaded by this app.')
        label.setWordWrap(True);layout.addWidget(label)
        self.text=QTextEdit(); self.text.setAcceptRichText(False)
        self.text.setPlaceholderText('# Song title\nSource: source URL or local licensed score\nVersion: production and language\nPaste reference lyrics here')
        if references:self.text.setPlainText(json.dumps(dict(schema='splytnyt-lyrics/1',references=references),indent=2,ensure_ascii=False))
        layout.addWidget(self.text)
        row=QHBoxLayout();layout.addLayout(row)
        load=QPushButton('Import lyrics…');load.clicked.connect(self.load);row.addWidget(load)
        save=QPushButton('Match local transcript');save.clicked.connect(self.validate);row.addWidget(save)
        close=QPushButton('Cancel');close.clicked.connect(self.reject);row.addWidget(close)
        self.references=[]
    def load(self):
        path,_=QFileDialog.getOpenFileName(self,'Import reference lyrics','','Lyrics (*.txt *.json);;All files (*)')
        if path:
            try:
                with open(path,encoding='utf-8-sig') as f:self.text.setPlainText(f.read())
            except (OSError,UnicodeError) as exc:QMessageBox.warning(self,'Lyrics',str(exc))
    def validate(self):
        try:self.references=parse_references(self.text.toPlainText())
        except (ValueError,TypeError,KeyError) as exc:
            QMessageBox.warning(self,'Lyrics',str(exc));return
        self.accept()


class LyricReviewDialog(QDialog):
    def __init__(self, report, window):
        super().__init__(window); self.setWindowTitle('Lyric-assisted musical-number review');self.resize(1100,680)
        self.window=window; self.report=report; layout=QVBoxLayout(self)
        label=QLabel('Proposals only. Text anchors and forced alignment do not verify identity. Preview anchors, gaps and intro/tail by ear.\n'
                     'Applying a proposed grouping is undoable (Ctrl+Z) and preserves locked or manually edited tracks. Double-click a row to seek without autoplay.')
        label.setWordWrap(True);layout.addWidget(label)
        self.tree=QTreeWidget();self.tree.setHeaderLabels(['Reference / evidence','Status','Start (s)','End (s)','Details'])
        layout.addWidget(self.tree)
        self._all_proposals=[]
        rate=window.session.project.sample_rate() or 48000
        for result in report.get('results',[]):
            root=QTreeWidgetItem(self.tree,[result['title'],'unmatched' if result['unmatched'] else 'unverified','','',result['provenance']+' | '+result['version']])
            root.setToolTip(4,result.get('scope',''))
            for occurrence in result['occurrences']:
                extent=occurrence.get('boundaries')
                prop_start=extent['start_frame'] if extent and extent.get('end_frame',0)>extent.get('start_frame',0) else round(occurrence['start']*rate)
                prop_end=extent['end_frame'] if extent and extent.get('end_frame',0)>extent.get('start_frame',0) else round(occurrence['end']*rate)
                missing_roles=list(occurrence.get('missing_roles') or [])
                intro_unres=bool(extent.get('intro_unresolved',True)) if extent else True
                tail_unres=bool(extent.get('tail_unresolved',True)) if extent else True
                is_uncertain=(
                    occurrence.get('status')!='multi-anchor proposal'
                    or bool(missing_roles)
                    or intro_unres
                    or tail_unres
                )
                proposal=dict(
                    title=result['title'],
                    start_frame=prop_start,
                    end_frame=prop_end,
                    status=occurrence['status'],
                    provenance=result.get('provenance',''),
                    missing_roles=missing_roles,
                    intro_unresolved=intro_unres,
                    tail_unresolved=tail_unres,
                    uncertain=is_uncertain,
                )
                self._all_proposals.append(proposal)
                item=QTreeWidgetItem(root,['Number occurrence',occurrence['status'],f"{occurrence['start']:.2f}",f"{occurrence['end']:.2f}",
                    'Missing roles: '+', '.join(occurrence['missing_roles'])+' | alignment: '+occurrence['alignment']['status']])
                item.setData(0,Qt.UserRole,(occurrence['start'],occurrence['end']))
                item.setData(0,Qt.UserRole+1,proposal)
                for a in occurrence['anchors']:
                    child=QTreeWidgetItem(item,[a['role']+': '+a['text'],'text candidate',f"{a['start']:.2f}",f"{a['end']:.2f}",
                        f"similarity {a['score']:.3f}; {a['timing']} timing; ASR: {a['transcript']}"])
                    child.setData(0,Qt.UserRole,(a['start'],a['end']))
                    child.setData(0,Qt.UserRole+1,proposal)
                for gap in occurrence['gaps']:
                    child=QTreeWidgetItem(item,['Uncertain gap','needs listening',f"{gap['start']:.2f}",f"{gap['end']:.2f}",gap['reason']])
                    child.setData(0,Qt.UserRole,(gap['start'],gap['end']))
                    child.setData(0,Qt.UserRole+1,proposal)
                for record in occurrence['alignment'].get('records',[]):
                    aligned=record['result']
                    child=QTreeWidgetItem(item,['Forced alignment: '+record['anchor'],aligned['status'],
                        f"{record['start']:.2f}",f"{record['end']:.2f}",
                        f"coverage {aligned.get('coverage',0):.3f}; "+aligned.get('reason','')])
                    child.setData(0,Qt.UserRole,(record['start'],record['end']))
                    child.setData(0,Qt.UserRole+1,proposal)
                if extent:
                    child=QTreeWidgetItem(item,['Local audio extent (provisional)',extent['status'],f"{extent['start_frame']/rate:.2f}",f"{extent['end_frame']/rate:.2f}",extent['reason']])
                    child.setData(0,Qt.UserRole,(extent['start_frame']/rate,extent['end_frame']/rate))
                    child.setData(0,Qt.UserRole+1,proposal)
        self.tree.expandToDepth(1);self.tree.setColumnWidth(0,320);self.tree.setColumnWidth(1,150)
        self.tree.itemDoubleClicked.connect(self.seek)
        self.details=QTextEdit();self.details.setReadOnly(True);self.details.setMaximumHeight(110)
        self.details.setPlaceholderText('Select evidence to read its full source, limitations or failure reason.')
        layout.addWidget(self.details)
        self.tree.currentItemChanged.connect(self.describe)
        row=QHBoxLayout();layout.addLayout(row)
        play=QPushButton('Play evidence range');play.clicked.connect(self.play);row.addWidget(play)
        stop=QPushButton('Stop');stop.clicked.connect(window.stop_playback);row.addWidget(stop)
        self.apply_button=QPushButton('Apply proposed grouping')
        self.apply_button.setProperty('primary',True)
        self.apply_button.setToolTip('Group the selected proposal (or all multi-anchor proposals) on the timeline. Undoable with Ctrl+Z; locked tracks are preserved.')
        self.apply_button.setEnabled(bool(self._all_proposals))
        self.apply_button.clicked.connect(self.apply_grouping)
        row.addWidget(self.apply_button)
        export=QPushButton('Export evidence JSON…');export.clicked.connect(lambda:self.export(report));row.addWidget(export)
        close=QPushButton('Close');close.clicked.connect(self.accept);row.addWidget(close)
    def selected_proposals(self):
        item=self.tree.currentItem()
        if item is not None:
            prop=item.data(0,Qt.UserRole+1)
            if prop is not None:
                return [prop]
        return [p for p in self._all_proposals if not p.get('uncertain',False)]
    def apply_grouping(self):
        proposals=self.selected_proposals()
        if not proposals:
            QMessageBox.information(
                self,
                'Apply proposed grouping',
                'No complete, bounded proposals are available to apply automatically.\n'
                'Select a specific proposal row after listening if you want to apply an uncertain boundary explicitly.',
            )
            return
        uncertain=[p for p in proposals if p.get('uncertain',False)]
        allow_uncertain=False
        if uncertain:
            reasons=[]
            for p in uncertain:
                parts=[]
                if p.get('missing_roles'):
                    parts.append('missing roles: '+', '.join(p['missing_roles']))
                if p.get('intro_unresolved'):
                    parts.append('intro extent unresolved')
                if p.get('tail_unresolved'):
                    parts.append('tail extent unresolved')
                if p.get('status')!='multi-anchor proposal':
                    parts.append(p.get('status','partial proposal'))
                reasons.append(f"• {p.get('title','Untitled')}: {', '.join(parts) or 'uncertain boundaries'}")
            reply=QMessageBox.question(
                self,
                'Confirm uncertain boundaries',
                'The selected proposal has uncertain boundaries and will NOT be applied automatically:\n\n'
                + '\n'.join(reasons)
                + '\n\nApply it anyway as a Needs Review boundary (undoable with Ctrl+Z)?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply!=QMessageBox.Yes:
                return
            allow_uncertain=True
        result=self.window.apply_lyric_grouping(proposals,allow_uncertain=allow_uncertain)
        applied=result.get('applied',0)
        skipped=result.get('skipped_protected',0)
        skipped_unc=result.get('skipped_uncertain',0)
        if applied:
            msg=f'Applied {applied} proposed grouping(s) to the timeline. Press Ctrl+Z to undo.'
            if skipped:
                msg+=f'\n{skipped} proposal(s) were skipped because they overlap locked or manually edited tracks.'
            if skipped_unc:
                msg+=f'\n{skipped_unc} proposal(s) were skipped because their boundaries are uncertain.'
            QMessageBox.information(self,'Apply proposed grouping',msg)
        else:
            details=[]
            if skipped:
                details.append(f'{skipped} overlapped locked or manually edited tracks')
            if skipped_unc:
                details.append(f'{skipped_unc} had uncertain boundaries')
            suffix=(' ('+'; '.join(details)+')') if details else ''
            QMessageBox.information(self,'Apply proposed grouping','No groupings were applied'+suffix+'.')
    def interval(self):
        item=self.tree.currentItem()
        return item.data(0,Qt.UserRole) if item else None
    def describe(self,item,*args):
        self.details.setPlainText('\n'.join(item.text(i) for i in range(5))+'\n'+item.toolTip(4) if item else '')
    def seek(self,*args):
        interval=self.interval()
        if interval:
            self.window.stop_playback();self.window._on_cursor_moved(round(interval[0]*self.window.session.project.sample_rate()))
    def play(self):
        interval=self.interval()
        if interval:
            rate=self.window.session.project.sample_rate()
            self.window._play_range(round(interval[0]*rate),round(interval[1]*rate),'Previewing unverified lyric evidence')
    def export(self,report):
        path,_=QFileDialog.getSaveFileName(self,'Export lyric evidence','','JSON (*.json)')
        if path:
            try:
                with open(path,'w',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False)
            except OSError as exc:QMessageBox.warning(self,'Export evidence',str(exc))
