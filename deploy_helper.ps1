# ===================================================================
#  VNeTraffic - Deploy Helper (PowerShell)
#  Duoc goi tu deploy.bat - Khong chay truc tiep
# ===================================================================
param(
    [string]$Action,
    [string]$AppDir,
    [string]$Version,
    [string]$BuildNumber
)

switch ($Action) {
    "get-ip" {
        # Tim IP WiFi / Ethernet cua may
        $adapters = Get-NetIPAddress -AddressFamily IPv4 | Where-Object {
            $_.InterfaceAlias -match 'Wi-Fi|WiFi|Wireless|Ethernet|LAN' -and
            $_.PrefixOrigin -eq 'Dhcp'
        }
        $ip = ($adapters | Select-Object -First 1).IPAddress
        if (-not $ip) {
            $fallback = Get-NetIPAddress -AddressFamily IPv4 | Where-Object {
                $_.IPAddress -notlike '127.*' -and
                $_.IPAddress -notlike '169.*' -and
                $_.PrefixOrigin -ne 'WellKnown'
            }
            $ip = ($fallback | Select-Object -First 1).IPAddress
        }
        if ($ip) { Write-Output $ip } else { Write-Output '0.0.0.0' }
    }

    "set-ip" {
        # Ghi IP vao api_service.dart
        $file = Join-Path $AppDir "lib\services\api_service.dart"
        $content = Get-Content $file -Raw
        $content = $content -replace "static String serverIp = '[^']*'", "static String serverIp = '$Version'"
        Set-Content $file -Value $content -NoNewline -Encoding UTF8
        Write-Output "OK"
    }

    "set-version" {
        # Ghi version vao pubspec.yaml
        $file = Join-Path $AppDir "pubspec.yaml"
        $content = Get-Content $file -Raw
        $content = $content -replace '(?m)^version: .+', "version: $Version+$BuildNumber"
        Set-Content $file -Value $content -NoNewline -Encoding UTF8
        Write-Output "OK"
    }

    "increment-version" {
        # Tu dong tang Patch version va Build number trong pubspec.yaml
        $file = Join-Path $AppDir "pubspec.yaml"
        $content = Get-Content $file -Raw
        if ($content -match '(?m)^version:\s*([0-9]+)\.([0-9]+)\.([0-9]+)\+([0-9]+)') {
            $major = $Matches[1]
            $minor = $Matches[2]
            $patch = [int]$Matches[3] + 1
            $build = [int]$Matches[4] + 1
            
            $newVer = "$major.$minor.$patch"
            $content = $content -replace '(?m)^version: .+', "version: $newVer+$build"
            Set-Content $file -Value $content -NoNewline -Encoding UTF8
            Write-Output "${newVer}_${build}"
        } elseif ($content -match '(?m)^version:\s*([0-9\.]+)\+([0-9]+)') {
            $ver = $Matches[1]
            $build = [int]$Matches[2] + 1
            $content = $content -replace '(?m)^version: .+', "version: $ver+$build"
            Set-Content $file -Value $content -NoNewline -Encoding UTF8
            Write-Output "${ver}_${build}"
        } else {
            # Mac dinh neu khong tim thay matching pattern
            Write-Output "1.0.0_1"
        }
    }
}
