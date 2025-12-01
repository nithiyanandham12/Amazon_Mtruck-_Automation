# PowerShell Script to Push MTurk AI Assistant to Hugging Face Spaces
# Uses your HF access token

param(
    [Parameter(Mandatory=$true)]
    [string]$HFUsername,
    
    [Parameter(Mandatory=$true)]
    [string]$SpaceName,
    
    [Parameter(Mandatory=$false)]
    [string]$AccessToken = $env:HF_TOKEN
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 Pushing MTurk AI Assistant to Hugging Face Spaces..." -ForegroundColor Green
Write-Host "Space: $HFUsername/$SpaceName" -ForegroundColor Cyan

# Set working directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Step 1: Check if Space repo exists locally
$spacePath = Join-Path (Split-Path -Parent $scriptDir) $SpaceName

if (-not (Test-Path $spacePath)) {
    Write-Host "`n📥 Cloning Space repository..." -ForegroundColor Yellow
    $cloneUrl = "https://$HFUsername`:$AccessToken@huggingface.co/spaces/$HFUsername/$SpaceName"
    
    try {
        Set-Location (Split-Path -Parent $scriptDir)
        git clone $cloneUrl
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Failed to clone. Creating Space first at: https://huggingface.co/new-space" -ForegroundColor Red
            Write-Host "   Choose 'Gradio' SDK and name it: $SpaceName" -ForegroundColor Yellow
            exit 1
        }
        Write-Host "✅ Repository cloned successfully!" -ForegroundColor Green
    } catch {
        Write-Host "❌ Error cloning repository: $_" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "✅ Space repository already exists locally." -ForegroundColor Green
}

# Step 2: Copy files to the cloned repository
Write-Host "`n📋 Copying files to Space repository..." -ForegroundColor Yellow
Set-Location $spacePath

$filesToCopy = @(
    "app.py",
    "requirements.txt",
    "README.md",
    "Dockerfile",
    ".gitignore",
    "saimese_mturk.pth"
)

$copiedCount = 0
foreach ($file in $filesToCopy) {
    $sourceFile = Join-Path $scriptDir $file
    if (Test-Path $sourceFile) {
        Copy-Item -Path $sourceFile -Destination . -Force
        Write-Host "  ✓ Copied $file" -ForegroundColor Gray
        $copiedCount++
    } else {
        Write-Host "  ⚠ File not found: $file" -ForegroundColor Yellow
    }
}

Write-Host "✅ Copied $copiedCount files" -ForegroundColor Green

# Step 3: Configure git
Write-Host "`n⚙️ Configuring git..." -ForegroundColor Yellow
git config user.name $HFUsername
git config user.email "$HFUsername@users.noreply.huggingface.co"

# Step 4: Check git status
Write-Host "`n📊 Checking git status..." -ForegroundColor Yellow
git status

# Step 5: Add files
Write-Host "`n📝 Staging files..." -ForegroundColor Yellow
git add .

# Step 6: Commit
Write-Host "`n💾 Committing changes..." -ForegroundColor Yellow
$commitMessage = "Add MTurk AI Assistant application with Gemini integration"
git commit -m $commitMessage

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️ No changes to commit or commit failed." -ForegroundColor Yellow
} else {
    Write-Host "✅ Changes committed!" -ForegroundColor Green
}

# Step 7: Push to Hugging Face
Write-Host "`n📤 Pushing to Hugging Face Spaces..." -ForegroundColor Yellow
Write-Host "   (Using access token for authentication)" -ForegroundColor Gray

# Set up credential helper
$env:GIT_TERMINAL_PROMPT = "0"
git remote set-url origin "https://$HFUsername`:$AccessToken@huggingface.co/spaces/$HFUsername/$SpaceName"

try {
    git push
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ Successfully pushed to Hugging Face Spaces!" -ForegroundColor Green
        Write-Host "`n🌐 Your Space will be available at:" -ForegroundColor Cyan
        Write-Host "   https://huggingface.co/spaces/$HFUsername/$SpaceName" -ForegroundColor White
        Write-Host "`n⚠️ IMPORTANT: Set GOOGLE_API_KEY in Space settings:" -ForegroundColor Yellow
        Write-Host "   1. Go to: https://huggingface.co/spaces/$HFUsername/$SpaceName/settings" -ForegroundColor White
        Write-Host "   2. Click 'Variables and secrets'" -ForegroundColor White
        Write-Host "   3. Add secret: GOOGLE_API_KEY = your Gemini API key" -ForegroundColor White
    } else {
        Write-Host "❌ Push failed. Check the error above." -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Error pushing: $_" -ForegroundColor Red
}

Write-Host "`n✨ Done!" -ForegroundColor Green

