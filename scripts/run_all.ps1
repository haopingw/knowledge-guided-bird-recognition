# Full test-set experiment (Windows PowerShell). Resumable: re-run to continue.
# Requires $env:KGBR_CUB_ROOT and a Python with the package installed ($env:KGBR_PY, default "python").
# Note: we check exit codes instead of $ErrorActionPreference="Stop", because Windows PowerShell 5.1
# turns any stderr line (e.g. harmless torch warnings) into a terminating error under "Stop".
$py = if ($env:KGBR_PY) { $env:KGBR_PY } else { "python" }
$env:TQDM_DISABLE = "1"
$steps = @(
    @("-m", "kgbr.run", "--split", "test", "--conditions", "A,B,C,D,E", "--tag", "main"),
    @("-m", "kgbr.run", "--split", "test", "--occlusion", "--conditions", "A,C", "--tag", "occlusion"),
    @("-m", "kgbr.probe"),
    @("-m", "kgbr.analyze", "full")
)
foreach ($s in $steps) {
    & $py -W ignore @s
    if ($LASTEXITCODE -ne 0) { Write-Output "FAILED: $($s -join ' ') (exit $LASTEXITCODE)"; exit $LASTEXITCODE }
}
