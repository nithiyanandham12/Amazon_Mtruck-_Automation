# Deploying to Hugging Face Spaces

## Option 1: Using Gradio SDK (Recommended - Simpler)

Since this is a Gradio app, you can deploy it directly without Docker:

1. **Create a new Space on Hugging Face:**
   - Go to https://huggingface.co/new-space
   - Choose "Gradio" as SDK
   - Name your space (e.g., `YourUsername/mturk-ai-assistant`)

2. **Clone the Space:**
```bash
git clone https://huggingface.co/spaces/YourUsername/YourSpaceName
cd YourSpaceName
```

3. **Copy your files:**
```bash
# Copy from your local directory
cp -r ../mturk-ai-Dual-Ensemble-Model/* .
```

4. **Set up Git:**
```bash
git config user.name "YourName"
git config user.email "your.email@example.com"
```

5. **Commit and push:**
```bash
git add .
git commit -m "Add MTurk AI Assistant app"
git push
```

6. **Set Environment Variables:**
   - Go to your Space settings: https://huggingface.co/spaces/YourUsername/YourSpaceName/settings
   - Add secret: `GOOGLE_API_KEY` with your Gemini API key

## Option 2: Using Docker (As per your instructions)

1. **Create a Docker Space:**
   - Go to https://huggingface.co/new-space
   - Choose "Docker" as SDK
   - Name your space

2. **Clone the Space:**
```powershell
# Generate access token from: https://huggingface.co/settings/tokens
git clone https://huggingface.co/spaces/YourUsername/YourSpaceName
cd YourSpaceName
```

3. **Install HF CLI (if needed):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

4. **Copy your files:**
```powershell
# Copy all files from mturk-ai-Dual-Ensemble-Model to the cloned repo
Copy-Item -Path ..\mturk-ai-Dual-Ensemble-Model\* -Destination . -Recurse -Force
```

5. **Commit and push:**
```bash
git add requirements.txt app.py Dockerfile README.md saimese_mturk.pth
git commit -m "Add MTurk AI Assistant application"
git push
```

6. **Set Environment Variables:**
   - Go to your Space settings
   - Add secret: `GOOGLE_API_KEY` with your Gemini API key

## Important Notes

- **Port**: Your app must listen on port 7860 (Gradio does this automatically)
- **Model File**: Make sure `saimese_mturk.pth` is committed (it's ~704MB, so it may take time)
- **API Key**: Set `GOOGLE_API_KEY` in Space secrets for Gemini to work
- **Build Time**: First build may take 10-15 minutes due to model downloads

## Troubleshooting

- If build fails, check logs in Space settings
- Ensure all dependencies are in requirements.txt
- Verify Dockerfile CMD is correct
- Check that port 7860 is exposed


