# How to Run the MTurk AI Assistant Frontend

## Prerequisites

1. **Python 3.8+** installed
2. **Google Gemini API Key** (required for AI features)

## Setup Steps

### 1. Install Dependencies

```bash
cd mturk-ai-Dual-Ensemble-Model
pip install -r requirements.txt
```

### 2. Set Environment Variables

**Windows PowerShell:**
```powershell
$env:GOOGLE_API_KEY="your-gemini-api-key-here"
```

**Windows CMD:**
```cmd
set GOOGLE_API_KEY=your-gemini-api-key-here
```

**Linux/Mac:**
```bash
export GOOGLE_API_KEY="your-gemini-api-key-here"
```

**Or create a `.env` file** (if using python-dotenv):
```
GOOGLE_API_KEY=your-gemini-api-key-here
```

### 3. Ensure Model File Exists

Make sure `saimese_mturk.pth` is in the same directory as `app.py`. It should have been downloaded automatically.

### 4. Run the Application

Simply run:
```bash
python app.py
```

The Gradio interface will launch automatically and display a URL (usually `http://127.0.0.1:7860`).

Open that URL in your browser to access the web interface.

## Features Available in the UI

1. **Image Evaluation**: Compare two response images against user instructions
2. **Style Annotation**: Evaluate if images match style guidelines (Hell Yes/Yes/No/Hell No)
3. **Prompt Evaluation**: Compare two prompts to see which better describes an editing process

## Troubleshooting

- **"No API key found"**: Make sure `GOOGLE_API_KEY` environment variable is set
- **Model loading errors**: Ensure `saimese_mturk.pth` is in the correct directory
- **Port already in use**: Change the port in `app.py` by modifying `demo.queue().launch(server_port=7861)`


