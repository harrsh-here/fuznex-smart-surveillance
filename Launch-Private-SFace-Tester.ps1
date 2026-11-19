$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# The existing desktop shortcut still points here, but now opens one GUI
# directly—there is no separate enrollment menu or repeated path prompt.
& python -m ai.spectra_console
exit $LASTEXITCODE
