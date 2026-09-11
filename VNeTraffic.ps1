[CmdletBinding()]
param(
    [ValidateSet('menu', 'setup', 'check', 'server', 'ngrok', 'app', 'build-apk', 'deploy-apk', 'firebase', 'help')]
    [string]$Action = 'menu',
    [int]$Port = 8000,
    [string]$Device = '',
    [string]$Changelog = 'Bug fixes and improvements',
    [switch]$ForceUpdate,
    [switch]$ResetVenv,
    [switch]$SkipNgrok,
    [string]$NgrokDomain = ''
)

$ErrorActionPreference = 'Stop'
$RepoRoot = $PSScriptRoot
$BackendDir = Join-Path $RepoRoot 'Detection Web\Web'
$BackendEntry = Join-Path $BackendDir 'app.py'
$AppDir = Join-Path $RepoRoot 'App\traffic_violation_app'
$ApiService = Join-Path $AppDir 'lib\services\api_service.dart'
$Pubspec = Join-Path $AppDir 'pubspec.yaml'
$ApkPath = Join-Path $AppDir 'build\app\outputs\flutter-apk\app-release.apk'
$Requirements = Join-Path $RepoRoot 'requirements.txt'
$VenvPython = if ($IsLinux -or $IsMacOS) {
    Join-Path $RepoRoot '.venv/bin/python'
} else {
    Join-Path $RepoRoot '.venv\Scripts\python.exe'
}

function Write-Title {
    param([string]$Text)
    Write-Host ''
    Write-Host ('=' * 68) -ForegroundColor DarkGray
    Write-Host "  $Text" -ForegroundColor Magenta
    Write-Host ('=' * 68) -ForegroundColor DarkGray
}

function Write-Step {
    param([string]$Text)
    Write-Host "`n[>] $Text" -ForegroundColor Cyan
}

function Assert-Command {
    param([string]$Name, [string]$InstallHint)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Khong tim thay '$Name'. $InstallHint"
    }
}

function Assert-File {
    param([string]$Path, [string]$Message)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Message`nPath: $Path"
    }
}

function Get-PythonCommand {
    if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
        return $VenvPython
    }
    throw "Chua co .venv. Chay: .\VNeTraffic.ps1 -Action setup"
}

function Get-LocalIPv4 {
    $candidates = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object {
            $_.IPAddress -notlike '127.*' -and
            $_.IPAddress -notlike '169.254.*' -and
            $_.InterfaceAlias -notmatch 'Loopback|WSL|vEthernet|Docker'
        } |
        Sort-Object -Property @{ Expression = { if ($_.PrefixOrigin -eq 'Dhcp') { 0 } else { 1 } } }

    $address = $candidates | Select-Object -First 1 -ExpandProperty IPAddress
    if (-not $address) {
        throw 'Khong tim thay IPv4 LAN/Wi-Fi. Hay ket noi mang va thu lai.'
    }
    return $address
}

function Set-AppServerAddress {
    param([string]$Address)
    Assert-File $ApiService 'Khong tim thay api_service.dart.'
    $content = Get-Content -LiteralPath $ApiService -Raw
    $updated = $content -replace "static String serverIp = '[^']*'", "static String serverIp = '$Address'"
    if ($updated -eq $content -and $content -notmatch [regex]::Escape("serverIp = '$Address'")) {
        throw 'Khong tim thay bien serverIp de cap nhat.'
    }
    Set-Content -LiteralPath $ApiService -Value $updated -NoNewline -Encoding UTF8
}

function Update-AppVersion {
    Assert-File $Pubspec 'Khong tim thay pubspec.yaml.'
    $content = Get-Content -LiteralPath $Pubspec -Raw
    if ($content -notmatch '(?m)^version:\s*(\d+)\.(\d+)\.(\d+)\+(\d+)\s*$') {
        throw 'Version trong pubspec.yaml phai co dang x.y.z+build.'
    }

    $newVersion = "$($Matches[1]).$($Matches[2]).$([int]$Matches[3] + 1)"
    $newBuild = [int]$Matches[4] + 1
    $updated = $content -replace '(?m)^version:\s*.+$', "version: $newVersion+$newBuild"
    Set-Content -LiteralPath $Pubspec -Value $updated -NoNewline -Encoding UTF8
    return [PSCustomObject]@{ Version = $newVersion; Build = $newBuild }
}

function Resolve-Ngrok {
    $bundled = Join-Path $RepoRoot 'ngrok_bin\ngrok.exe'
    if (Test-Path -LiteralPath $bundled -PathType Leaf) { return $bundled }
    $command = Get-Command ngrok -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    throw 'Khong tim thay ngrok. Cai dat tai https://ngrok.com/download hoac dat ngrok.exe trong ngrok_bin/.'
}

function Invoke-Setup {
    Write-Title 'VNeTraffic - Cai dat backend'
    Assert-Command python 'Cai Python 3.10+ va chon Add Python to PATH.'
    $venvDir = Join-Path $RepoRoot '.venv'
    if ($ResetVenv -and (Test-Path -LiteralPath $venvDir)) {
        $resolvedVenv = (Get-Item -LiteralPath $venvDir).FullName
        if ($resolvedVenv -ne (Join-Path $RepoRoot '.venv')) {
            throw "Tu choi xoa venv ngoai repo: $resolvedVenv"
        }
        Write-Step 'Xoa .venv cu theo yeu cau -ResetVenv'
        Remove-Item -LiteralPath $resolvedVenv -Recurse -Force
    }
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        Write-Step 'Tao moi truong ao .venv'
        & python -m venv (Join-Path $RepoRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Khong tao duoc .venv.' }
    }
    $python = Get-PythonCommand
    Write-Step 'Nang cap pip'
    & $python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'Khong nang cap duoc pip.' }
    Write-Step 'Cai Python dependencies'
    & $python -m pip install -r $Requirements
    if ($LASTEXITCODE -ne 0) { throw 'Cai dependencies that bai.' }
    Write-Host "`n[OK] Backend da san sang. Chay server bang option 3." -ForegroundColor Green
}

function Invoke-Check {
    Write-Title 'VNeTraffic - Kiem tra moi truong'
    $checks = @(
        @{ Name = 'Python venv'; Ok = Test-Path -LiteralPath $VenvPython; Detail = $VenvPython },
        @{ Name = 'Backend entry'; Ok = Test-Path -LiteralPath $BackendEntry; Detail = $BackendEntry },
        @{ Name = 'Default model'; Ok = Test-Path -LiteralPath (Join-Path $RepoRoot 'Detection Web\assets\model\yolo26_rbf.pt'); Detail = 'yolo26_rbf.pt' },
        @{ Name = 'Default video'; Ok = Test-Path -LiteralPath (Join-Path $RepoRoot 'Detection Web\assets\video\test_2_fixed.mp4'); Detail = 'test_2_fixed.mp4' },
        @{ Name = 'Flutter'; Ok = [bool](Get-Command flutter -ErrorAction SilentlyContinue); Detail = 'flutter --version' },
        @{ Name = 'Firebase CLI (optional)'; Ok = [bool](Get-Command firebase -ErrorAction SilentlyContinue); Detail = 'npm i -g firebase-tools' },
        @{ Name = 'Firebase Admin key (optional)'; Ok = Test-Path -LiteralPath (Join-Path $BackendDir 'serviceAccountKey.json'); Detail = 'FCM/Firestore can tat neu thieu' }
    )
    foreach ($check in $checks) {
        $state = if ($check.Ok) { '[OK] ' } else { '[--] ' }
        $color = if ($check.Ok) { 'Green' } else { 'Yellow' }
        Write-Host ($state + $check.Name + ' - ' + $check.Detail) -ForegroundColor $color
    }

    if (Test-Path -LiteralPath $VenvPython) {
        Write-Step 'Kiem tra import Python'
        & $VenvPython -c "import cv2, fastapi, firebase_admin, scipy, ultralytics, uvicorn; print('Python imports: OK')"
        if ($LASTEXITCODE -ne 0) { throw 'Thieu Python dependency. Chay option setup.' }
    }
}

function Invoke-Server {
    $python = Get-PythonCommand
    Assert-File $BackendEntry 'Khong tim thay backend app.py.'
    $oldPort = $env:VNETRAFFIC_PORT
    $oldNgrok = $env:VNETRAFFIC_ENABLE_NGROK
    try {
        $env:VNETRAFFIC_PORT = [string]$Port
        $env:VNETRAFFIC_ENABLE_NGROK = if ($SkipNgrok) { '0' } else { '1' }
        if (-not $SkipNgrok) {
            $ngrokDir = Join-Path $RepoRoot 'ngrok_bin'
            if (Test-Path -LiteralPath $ngrokDir) { $env:PATH = "$ngrokDir;$env:PATH" }
        }
        Write-Title "VNeTraffic Server - http://localhost:$Port"
        & $python $BackendEntry
        if ($LASTEXITCODE -ne 0) { throw "Server dung voi exit code $LASTEXITCODE." }
    } finally {
        $env:VNETRAFFIC_PORT = $oldPort
        $env:VNETRAFFIC_ENABLE_NGROK = $oldNgrok
    }
}

function Invoke-NgrokTunnel {
    $ngrok = Resolve-Ngrok
    Write-Title "Ngrok tunnel -> localhost:$Port"
    if ($NgrokDomain) {
        & $ngrok http "--url=$NgrokDomain" $Port
    } else {
        & $ngrok http $Port
    }
    if ($LASTEXITCODE -ne 0) { throw "Ngrok dung voi exit code $LASTEXITCODE." }
}

function Invoke-FlutterApp {
    Assert-Command flutter 'Cai Flutter SDK va chay flutter doctor.'
    Push-Location $AppDir
    try {
        Write-Step 'Tai Flutter packages'
        & flutter pub get
        if ($LASTEXITCODE -ne 0) { throw 'flutter pub get that bai.' }
        $args = @('run')
        if ($Device) { $args += @('-d', $Device) }
        Write-Step 'Khoi dong Flutter app'
        & flutter @args
        if ($LASTEXITCODE -ne 0) { throw "flutter run dung voi exit code $LASTEXITCODE." }
    } finally { Pop-Location }
}

function Invoke-BuildApk {
    Assert-Command flutter 'Cai Flutter SDK va Android toolchain, sau do chay flutter doctor.'
    Push-Location $AppDir
    try {
        Write-Title 'Build Android APK release'
        & flutter pub get
        if ($LASTEXITCODE -ne 0) { throw 'flutter pub get that bai.' }
        & flutter build apk --release
        if ($LASTEXITCODE -ne 0) { throw 'Build APK that bai.' }
        Assert-File $ApkPath 'Build xong nhung khong tim thay APK.'
        $sizeMb = [math]::Round((Get-Item -LiteralPath $ApkPath).Length / 1MB, 1)
        Write-Host "`n[OK] $ApkPath ($sizeMb MB)" -ForegroundColor Green
    } finally { Pop-Location }
}

function Invoke-DeployApk {
    $ip = Get-LocalIPv4
    Write-Title 'Build va phat hanh APK qua backend'
    $originalApiService = Get-Content -LiteralPath $ApiService -Raw
    $originalPubspec = Get-Content -LiteralPath $Pubspec -Raw
    try {
        Write-Step "Cap nhat server app: $ip`:$Port"
        Set-AppServerAddress -Address $ip
        $release = Update-AppVersion
        Write-Host "Version moi: $($release.Version)+$($release.Build)" -ForegroundColor Cyan
        Invoke-BuildApk

        $serverUrl = "http://$ip`:$Port"
        Write-Step "Upload APK toi $serverUrl"
        Assert-Command curl.exe 'Windows 10/11 thuong da co curl.exe.'
        $force = if ($ForceUpdate) { 'true' } else { 'false' }
        & curl.exe --fail --show-error -X POST "$serverUrl/api/app/upload-apk" `
            -F "file=@$ApkPath" `
            -F "version=$($release.Version)" `
            -F "build_number=$($release.Build)" `
            -F "changelog=$Changelog" `
            -F "force_update=$force"
        if ($LASTEXITCODE -ne 0) { throw 'Upload APK that bai. Dam bao backend dang chay va firewall cho phep port.' }
        Write-Host "`n[OK] Da phat hanh APK $($release.Version)+$($release.Build)." -ForegroundColor Green
    } catch {
        Set-Content -LiteralPath $ApiService -Value $originalApiService -NoNewline -Encoding UTF8
        Set-Content -LiteralPath $Pubspec -Value $originalPubspec -NoNewline -Encoding UTF8
        throw
    }
}

function Invoke-FirebaseRules {
    Assert-Command firebase 'Cai Node.js, chay npm install -g firebase-tools va firebase login.'
    Push-Location $RepoRoot
    try {
        Write-Title 'Deploy Firebase security rules'
        & firebase deploy --only 'firestore:rules,storage'
        if ($LASTEXITCODE -ne 0) { throw 'Firebase deploy that bai.' }
    } finally { Pop-Location }
}

function Show-Help {
    Write-Title 'VNeTraffic - Launcher'
    Write-Host @'
Usage:
  .\VNeTraffic.ps1                         Mo menu tuong tac
  .\VNeTraffic.ps1 -Action <action>        Chay truc tiep mot tac vu

Actions:
  setup        Tao .venv va cai Python dependencies
  check        Kiem tra Python, model, video, Flutter va Firebase
  server       Chay FastAPI dashboard (mac dinh kem ngrok)
  ngrok        Chi mo public tunnel toi backend
  app          Chay Flutter app tren device/emulator
  build-apk    Build APK release, khong upload
  deploy-apk   Tu tang version, build va upload APK len backend
  firebase     Deploy Firestore + Storage rules
  help         Hien huong dan nay

Options thuong dung:
  -Port 8000
  -SkipNgrok
  -Device <device-id>
  -Changelog "Noi dung cap nhat"
  -ForceUpdate
  -ResetVenv
  -NgrokDomain <your-domain.ngrok-free.app>

Vi du:
  .\VNeTraffic.ps1 -Action server -SkipNgrok
  .\VNeTraffic.ps1 -Action app -Device emulator-5554
  .\VNeTraffic.ps1 -Action deploy-apk -Changelog "Sua loi thong bao"
'@
}

function Show-Menu {
    while ($true) {
        Write-Title 'VNeTraffic - Chon tac vu'
        Write-Host '  1. Cai dat backend'
        Write-Host '  2. Kiem tra moi truong'
        Write-Host '  3. Chay server Web + AI'
        Write-Host '  4. Chay server khong Ngrok'
        Write-Host '  5. Mo Ngrok rieng'
        Write-Host '  6. Chay Flutter app'
        Write-Host '  7. Build APK release'
        Write-Host '  8. Build + upload APK'
        Write-Host '  9. Deploy Firebase rules'
        Write-Host '  H. Huong dan CLI'
        Write-Host '  0. Thoat'
        $choice = Read-Host "`nLua chon"
        try {
            switch ($choice.ToLowerInvariant()) {
                '1' { Invoke-Setup }
                '2' { Invoke-Check }
                '3' { Invoke-Server }
                '4' {
                    $previousSkipNgrok = $script:SkipNgrok
                    try { $script:SkipNgrok = $true; Invoke-Server }
                    finally { $script:SkipNgrok = $previousSkipNgrok }
                }
                '5' { Invoke-NgrokTunnel }
                '6' { Invoke-FlutterApp }
                '7' { Invoke-BuildApk }
                '8' { $script:Changelog = Read-Host 'Noi dung cap nhat'; if (-not $script:Changelog) { $script:Changelog = 'Bug fixes and improvements' }; Invoke-DeployApk }
                '9' { Invoke-FirebaseRules }
                'h' { Show-Help }
                '0' { return }
                default { Write-Host 'Lua chon khong hop le.' -ForegroundColor Yellow }
            }
        } catch {
            Write-Host "`n[LOI] $($_.Exception.Message)" -ForegroundColor Red
        }
        if ($choice -notin @('0', '3', '4', '5', '6')) {
            Read-Host "`nNhan Enter de quay lai menu" | Out-Null
        }
    }
}

try {
    switch ($Action) {
        'menu' { Show-Menu }
        'setup' { Invoke-Setup }
        'check' { Invoke-Check }
        'server' { Invoke-Server }
        'ngrok' { Invoke-NgrokTunnel }
        'app' { Invoke-FlutterApp }
        'build-apk' { Invoke-BuildApk }
        'deploy-apk' { Invoke-DeployApk }
        'firebase' { Invoke-FirebaseRules }
        'help' { Show-Help }
    }
} catch {
    Write-Host "`n[LOI] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
