# Step-by-Step: Push to Hugging Face Spaces

## Prerequisites

1. **Hugging Face Account**: Sign up at https://huggingface.co/join
2. **Access Token**: Generate from https://huggingface.co/settings/tokens (needs WRITE permission)

## Method 1: Gradio SDK (Easiest - Recommended)

### Step 1: Create New Space
1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name**: `mturk-ai-assistant` (or your preferred name)
   - **SDK**: Select **"Gradio"**
   - **Hardware**: CPU Basic (or GPU if needed)
   - Click **"Create Space"**

### Step 2: Clone Your New Space
```powershell
# Replace YourUsername with your HF username
git clone https://huggingface.co/spaces/YourUsername/mturk-ai-assistant
cd mturk-ai-assistant
```

### Step 3: Copy Files
```powershell
# From the mturk-ai-Dual-Ensemble-Model directory
Copy-Item -Path "E:\Amazon_MTruck\mturk-ai-Dual-Ensemble-Model\*" -Destination . -Recurse -Force -Exclude ".git"
```

### Step 4: Configure Git
```powershell
git config user.name "YourName"
git config user.email "your.email@example.com"
```

### Step 5: Commit and Push
```powershell
git add .
git commit -m "Add MTurk AI Assistant with Gemini integration"
git push
```

### Step 6: Set API Key Secret
1. Go to: https://huggingface.co/spaces/YourUsername/mturk-ai-assistant/settings
2. Click **"Variables and secrets"**
3. Add new secret:
   - **Key**: `GOOGLE_API_KEY`
   - **Value**: Your Gemini API key
4. Click **"Save"**

## Method 2: Docker SDK (As per your instructions)

### Step 1: Create Docker Space
1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name**: `mturk-ai-assistant`
   - **SDK**: Select **"Docker"**
   - Click **"Create Space"**

### Step 2: Clone Space
```powershell
git clone https://huggingface.co/spaces/YourUsername/mturk-ai-assistant
cd mturk-ai-assistant
```

### Step 3: Install HF CLI (if needed)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

### Step 4: Copy Files
```powershell
Copy-Item -Path "E:\Amazon_MTruck\mturk-ai-Dual-Ensemble-Model\*" -Destination . -Recurse -Force -Exclude ".git"
```

### Step 5: Commit and Push
```powershell
git add requirements.txt app.py Dockerfile README.md saimese_mturk.pth .gitignore
git commit -m "Add MTurk AI Assistant application"
git push
```

### Step 6: Set API Key Secret
Same as Method 1, Step 6

## Important Notes

⚠️ **Model File Size**: `saimese_mturk.pth` is ~704MB. First push may take time.

⚠️ **Build Time**: First build on HF Spaces takes 10-15 minutes (downloads models).

⚠️ **Port**: App must listen on port 7860 (already configured in app.py).

✅ **Check Status**: Monitor build logs at: `https://huggingface.co/spaces/YourUsername/mturk-ai-assistant`

## Troubleshooting

- **Authentication Error**: Use access token as password when prompted
- **Build Fails**: Check logs in Space settings → Logs tab
- **API Key Not Working**: Verify secret name is exactly `GOOGLE_API_KEY`
- **Port Error**: Ensure app listens on 0.0.0.0:7860 (Gradio does this automatically)


