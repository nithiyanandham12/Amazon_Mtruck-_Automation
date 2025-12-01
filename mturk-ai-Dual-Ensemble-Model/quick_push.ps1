# Quick Push Script - Interactive version
# This script will guide you through pushing to Hugging Face Spaces
# Set HF_TOKEN environment variable or enter token when prompted

$AccessToken = $env:HF_TOKEN
if ([string]::IsNullOrWhiteSpace($AccessToken)) {
    $AccessToken = Read-Host "Enter your Hugging Face access token (or set HF_TOKEN env var)"
}

Write-Host "`n🚀 MTurk AI Assistant - Hugging Face Spaces Deployment" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan

# Get username
$HFUsername = Read-Host "`nEnter your Hugging Face username"
if ([string]::IsNullOrWhiteSpace($HFUsername)) {
    Write-Host "❌ Username is required!" -ForegroundColor Red
    exit 1
}

# Get space name
$SpaceName = Read-Host "Enter Space name (or press Enter for 'mturk-ai-assistant')"
if ([string]::IsNullOrWhiteSpace($SpaceName)) {
    $SpaceName = "mturk-ai-assistant"
}

Write-Host "`n📋 Summary:" -ForegroundColor Yellow
Write-Host "   Username: $HFUsername" -ForegroundColor White
Write-Host "   Space: $SpaceName" -ForegroundColor White
Write-Host "   URL: https://huggingface.co/spaces/$HFUsername/$SpaceName" -ForegroundColor White

$confirm = Read-Host "`nContinue? (Y/N)"
if ($confirm -ne "Y" -and $confirm -ne "y") {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit 0
}

# Check if Space exists
Write-Host "`n🔍 Checking if Space exists..." -ForegroundColor Yellow
$spaceUrl = "https://huggingface.co/spaces/$HFUsername/$SpaceName"
try {
    $response = Invoke-WebRequest -Uri $spaceUrl -Method Head -ErrorAction SilentlyContinue
    Write-Host "✅ Space exists!" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Space not found. You need to create it first:" -ForegroundColor Yellow
    Write-Host "   1. Go to: https://huggingface.co/new-space" -ForegroundColor White
    Write-Host "   2. Choose 'Gradio' as SDK" -ForegroundColor White
    Write-Host "   3. Name it: $SpaceName" -ForegroundColor White
    Write-Host "   4. Click 'Create Space'" -ForegroundColor White
    Write-Host "`nPress Enter after creating the Space to continue..." -ForegroundColor Cyan
    Read-Host
}

# Set working directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$parentDir = Split-Path -Parent $scriptDir
$spacePath = Join-Path $parentDir $SpaceName

# Clone if needed
if (-not (Test-Path $spacePath)) {
    Write-Host "`n📥 Cloning Space repository..." -ForegroundColor Yellow
    Set-Location $parentDir
    $cloneUrl = "https://$HFUsername`:$AccessToken@huggingface.co/spaces/$HFUsername/$SpaceName"
    git clone $cloneUrl
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to clone. Please check your Space name and try again." -ForegroundColor Red
        exit 1
    }
}

# Copy files
Write-Host "`n📋 Copying files..." -ForegroundColor Yellow
Set-Location $spacePath

$files = @("app.py", "requirements.txt", "README.md", "Dockerfile", ".gitignore", "saimese_mturk.pth")
foreach ($file in $files) {
    $src = Join-Path $scriptDir $file
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination . -Force
        Write-Host "  ✓ $file" -ForegroundColor Gray
    }
}

# Configure git
git config user.name $HFUsername
git config user.email "$HFUsername@users.noreply.huggingface.co"

# Commit and push
Write-Host "`n💾 Committing changes..." -ForegroundColor Yellow
git add .
git commit -m "Add MTurk AI Assistant application"

Write-Host "`n📤 Pushing to Hugging Face..." -ForegroundColor Yellow
git remote set-url origin "https://$HFUsername`:$AccessToken@huggingface.co/spaces/$HFUsername/$SpaceName"
git push

Write-Host "`n✅ Done! Your Space is being built." -ForegroundColor Green
Write-Host "`n⚠️ Don't forget to set GOOGLE_API_KEY in Space settings!" -ForegroundColor Yellow

