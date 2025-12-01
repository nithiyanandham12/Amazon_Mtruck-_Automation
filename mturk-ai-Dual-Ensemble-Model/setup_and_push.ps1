# PowerShell Script to Push MTurk AI Assistant to Hugging Face Spaces
# Before running: 
# 1. Create a Space at https://huggingface.co/new-space (choose Gradio SDK)
# 2. Get your HF username and Space name
# 3. Generate access token from https://huggingface.co/settings/tokens

param(
    [Parameter(Mandatory=$true)]
    [string]$HFUsername,
    
    [Parameter(Mandatory=$true)]
    [string]$SpaceName,
    
    [Parameter(Mandatory=$false)]
    [string]$AccessToken
)

Write-Host "🚀 Setting up Hugging Face Space deployment..." -ForegroundColor Green

# Step 1: Navigate to parent directory
$parentDir = Split-Path -Parent $PSScriptRoot
Set-Location $parentDir

# Step 2: Clone the HF Space (if not already cloned)
$spacePath = Join-Path $parentDir $SpaceName
if (-not (Test-Path $spacePath)) {
    Write-Host "📥 Cloning Space repository..." -ForegroundColor Yellow
    if ($AccessToken) {
        $cloneUrl = "https://$HFUsername`:$AccessToken@huggingface.co/spaces/$HFUsername/$SpaceName"
    } else {
        $cloneUrl = "https://huggingface.co/spaces/$HFUsername/$SpaceName"
    }
    git clone $cloneUrl
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to clone repository. Please check your credentials." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "✅ Space repository already exists." -ForegroundColor Green
}

# Step 3: Copy files to the cloned repository
Write-Host "📋 Copying files to Space repository..." -ForegroundColor Yellow
Set-Location $spacePath

$sourceDir = Join-Path $parentDir "mturk-ai-Dual-Ensemble-Model"
$filesToCopy = @(
    "app.py",
    "requirements.txt",
    "README.md",
    "Dockerfile",
    ".gitignore",
    "saimese_mturk.pth"
)

foreach ($file in $filesToCopy) {
    $sourceFile = Join-Path $sourceDir $file
    if (Test-Path $sourceFile) {
        Copy-Item -Path $sourceFile -Destination . -Force
        Write-Host "  ✓ Copied $file" -ForegroundColor Gray
    } else {
        Write-Host "  ⚠ File not found: $file" -ForegroundColor Yellow
    }
}

# Step 4: Configure git (if needed)
Write-Host "⚙️ Configuring git..." -ForegroundColor Yellow
git config user.name $HFUsername
git config user.email "$HFUsername@users.noreply.huggingface.co"

# Step 5: Add and commit
Write-Host "📝 Staging files..." -ForegroundColor Yellow
git add .
git status

Write-Host "`n📤 Ready to commit and push!" -ForegroundColor Green
Write-Host "Run these commands manually:" -ForegroundColor Cyan
Write-Host "  git commit -m 'Add MTurk AI Assistant application'" -ForegroundColor White
Write-Host "  git push" -ForegroundColor White
Write-Host "`n💡 Don't forget to set GOOGLE_API_KEY in Space settings!" -ForegroundColor Yellow


