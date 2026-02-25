[Setup]
; --- 基本應用程式資訊 ---
AppName=Convnext_System
AppVersion=1.0.0
AppPublisher=PrefactorTech
LicenseFile=license.txt
OutputDir=InstallerOutput 
;  修改 4：安裝檔輸出的檔名 [cite: 116]
OutputBaseFilename=convnext_system_Setup_x64 
Compression=lzma
SolidCompression=yes
PrivilegesRequired=admin

ArchitecturesInstallIn64BitMode=x64
ArchitecturesAllowed=x64
;  修改 5：安裝在 C:\Program Files 裡的資料夾名稱 [cite: 116]
DefaultDirName={autopf64}\convnext_system 
DefaultGroupName=AI 視覺檢測系統 Pro
DisableDirPage=no

[Files]
Source: "vcredist_x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall
;  修改 6：這裡的資料夾名稱要跟 spec 檔的 COLLECT name 一致 [cite: 117]
Source: "dist\convnext_system\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
; 修改 7：捷徑指向的執行檔全改為 convnext_system.exe [cite: 118]
Name: "{group}\AI 視覺檢測系統 Pro"; Filename: "{app}\convnext_system.exe"; WorkingDir: "{app}"; IconFilename: "{app}\convnext_system.exe"
Name: "{autodesktop}\AI 視覺檢測系統 Pro"; Filename: "{app}\convnext_system.exe"; WorkingDir: "{app}"; IconFilename: "{app}\convnext_system.exe"
Name: "{group}\解除安裝"; Filename: "{uninstallexe}"

[Run]
Filename: "{tmp}\vcredist_x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "正在檢查並安裝 Microsoft VC++ 執行時期函式庫..."; Flags: waituntilterminated; Check: VCRedistNeedsInstall

; 修改 8：安裝完成後自動執行的程式名稱 [cite: 119, 120]
Filename: "{app}\convnext_system.exe"; Description: "啟動 AI 視覺檢測系統 Pro"; Flags: nowait postinstall skipifsilent

[Code]
// 檢查是否需要安裝 VC++ 的邏輯
function VCRedistNeedsInstall: Boolean;
var
  RegKey: String;
  Installed: Cardinal;
begin
  RegKey := 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64';
  if RegQueryDWordValue(HKEY_LOCAL_MACHINE, RegKey, 'Installed', Installed) then
  begin
    if Installed = 1 then
      Result := False
    else
      Result := True;
  end
  else
  begin
    Result := True;
  end;
end;