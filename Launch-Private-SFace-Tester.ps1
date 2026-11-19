$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "FuzNex private SFace tester" -ForegroundColor Cyan
Write-Host "Local only. Images, embeddings, model paths, and results are never exported by this launcher." -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Enroll or update a consenting participant from camera"
Write-Host "2. Check one consenting image against the private gallery"
Write-Host "3. Delete one participant's local template"
Write-Host "Q. Quit"
$action = Read-Host "Choose an option"
if ($action -eq "Q" -or $action -eq "q") { exit 0 }

$defaultGallery = Join-Path $PSScriptRoot "local_data\sface\private-gallery.json"
$gallery = Read-Host "Private gallery path [Enter for $defaultGallery]"
if ([string]::IsNullOrWhiteSpace($gallery)) { $gallery = $defaultGallery }

if ($action -eq "1") {
    $participant = Read-Host "Participant ID (participant_1, participant_2, or participant_3)"
    $consent = Read-Host "Type I_CONSENT to confirm this participant's informed consent"
    if ($consent -ne "I_CONSENT") { Write-Host "No consent confirmation: nothing was saved." -ForegroundColor Red; exit 1 }
    $scrfd = Read-Host "Full path to local SCRFD det_500m.onnx"
    $sface = Read-Host "Full path to local official SFace ONNX"
    $camera = Read-Host "Camera source [Enter for 0]"
    if ([string]::IsNullOrWhiteSpace($camera)) { $camera = "0" }
    $create = ""
    if (-not (Test-Path -LiteralPath $gallery)) { $create = "--create" }
    $args = @("-m", "ai.enrollment", "--gallery", $gallery, "--participant-id", $participant, "--consent", $consent, "--scrfd", $scrfd, "--sface", $sface, "--camera", $camera, "--samples", "5")
    if ($create) { $args += $create }
    & python @args
    exit $LASTEXITCODE
}

if ($action -eq "2") {
    $scrfd = Read-Host "Full path to local SCRFD det_500m.onnx"
    $sface = Read-Host "Full path to local official SFace ONNX"
    $image = Read-Host "Full path to one consenting check image"
    & python -m ai.console --local-diagnostic --scrfd $scrfd --sface $sface --gallery $gallery --private-recognition-image $image
    exit $LASTEXITCODE
}

if ($action -eq "3") {
    $participant = Read-Host "Participant ID to delete"
    $confirm = Read-Host "Repeat the same participant ID to confirm deletion"
    & python -m ai.enrollment --gallery $gallery --participant-id $participant --delete --confirm-delete $confirm
    exit $LASTEXITCODE
}

Write-Host "Unrecognised option; nothing was changed." -ForegroundColor Red
exit 1
