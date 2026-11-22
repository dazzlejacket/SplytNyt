; Inno Setup script for the SplytNyt Windows installer.
;
; Build order:
;     python scripts/build_windows.py --clean
;     "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" scripts\installer\SplytNyt.iss
;
; The PyInstaller build (dist\SplytNyt\) must exist before this runs.
; The installer is per-user: no administrator rights are needed, and it can be
; removed from Settings -> Apps -> Installed apps.

#define AppName "SplytNyt"
#define AppVersion "0.1.0-beta"
#define AppPublisher "dazzlejacket"
#define AppExeName "SplytNyt.exe"

[Setup]
AppId={{7C5A1E4B-9F3D-4A62-8C1E-5D7A9B2F3E61}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppComments=Split long audio recordings into named, tagged tracks.
AppPublisherURL=https://github.com/dazzlejacket/SplytNyt
AppSupportURL=https://github.com/dazzlejacket/SplytNyt/issues
AppUpdatesURL=https://github.com/dazzlejacket/SplytNyt/releases
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
UninstallDisplayName={#AppName} {#AppVersion}
OutputBaseFilename=SplytNyt-{#AppVersion}-setup
OutputDir=..\..\release
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
; Per-user install: no elevation, no admin rights, no machine-wide changes.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
UninstallDisplaySizeMB=300
SetupLogging=yes
VersionInfoVersion=0.1.0.0
VersionInfoDescription={#AppName} installer
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; The frozen application and everything it needs.
Source: "..\..\dist\SplytNyt\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Licence texts for the bundled components, including the LGPL ones.
Source: "..\..\licenses\*"; DestDir: "{app}\licenses"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\..\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\README.md"; DestDir: "{app}"; Flags: ignoreversion
; Uninstaller notes.
Source: "..\..\scripts\installer\uninstall-notes.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\{#AppName} on the web"; Filename: "https://github.com/dazzlejacket/SplytNyt"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
  { Refuse to install on anything that is not 64-bit Windows. }
  if not IsWin64 and not IsWin64CPU then
  begin
    MsgBox('SplytNyt needs a 64-bit version of Windows 10 or newer.', mbCriticalError, MB_OK);
    Result := False;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    { The user's own settings, projects, downloaded models and the optional
      ShazamIO engine live in %USERPROFILE%\.splytnyt. They are the user's
      data, not ours, so they are left alone on purpose. }
    DataDir := ExpandConstant('{userappdata}\.splytnyt');
    if DirExists(DataDir) then
    begin
      if MsgBox(
        'SplytNyt has been removed.' + #13#10 + #13#10 +
        'Your settings, projects, downloaded models and the optional ShazamIO ' +
        'engine are still in:' + #13#10 + #13#10 + DataDir + #13#10 + #13#10 +
        'Do you want to delete that folder as well?',
        mbConfirmation, MB_YESNO) = IDYES then
      begin
        DelTree(DataDir, True, True, True);
      end;
    end;
  end;
end;
