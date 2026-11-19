param(
    [string]$Python = '',
    [switch]$NoShortcuts
)

$ErrorActionPreference = 'Stop'
$app = Join-Path $PSScriptRoot 'ChatDFT.exe'
$requirements = Join-Path $PSScriptRoot 'requirements.txt'
if (-not (Test-Path -LiteralPath $app -PathType Leaf)) {
    throw 'Extract the complete ChatDFT archive before installing.'
}

if (-not $Python) {
    $environment = Join-Path $PSScriptRoot '.venv'
    $Python = Join-Path $environment 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
        if (-not $launcher) {
            throw 'Install 64-bit Python 3.13 or pass -Python with an existing compatible environment.'
        }
        & $launcher.Source -3.13-64 -m venv $environment
        if ($LASTEXITCODE) { throw 'Could not create the ChatDFT Python environment.' }
    }
    & $Python -m pip install -r $requirements
    if ($LASTEXITCODE) { throw 'Could not install the ChatDFT runtime dependencies.' }
}

$Python = (Resolve-Path -LiteralPath $Python).Path
& $Python (Join-Path $PSScriptRoot 'configure_runtime.py')
if ($LASTEXITCODE) {
    throw 'The ChatDFT runtime could not be configured. Check Python and the package dependencies.'
}

if (-not $NoShortcuts) {
    $shell = New-Object -ComObject WScript.Shell
    foreach ($folder in @('Desktop', 'Programs')) {
        $directory = [Environment]::GetFolderPath($folder)
        if (-not $directory) { continue }
        $path = Join-Path $directory 'ChatDFT.lnk'
        $shortcut = $shell.CreateShortcut($path)
        $backup = $path + '.pre-standalone.bak'
        if ((Test-Path -LiteralPath $path) -and $shortcut.TargetPath -ne $app -and
            -not (Test-Path -LiteralPath $backup)) {
            Copy-Item -LiteralPath $path -Destination $backup
        }
        $shortcut.TargetPath = $app
        $shortcut.Arguments = ''
        $shortcut.WorkingDirectory = $PSScriptRoot
        $shortcut.IconLocation = $app + ',0'
        $shortcut.Description = 'ChatDFT 5.9 independent molecular workbench'
        $shortcut.Save()
        $saved = $shell.CreateShortcut($path)
        if ($saved.TargetPath -ne $app) { throw "Incorrect shortcut target: $path" }
        Write-Output ("ChatDFT shortcut: " + $path)
    }
}
