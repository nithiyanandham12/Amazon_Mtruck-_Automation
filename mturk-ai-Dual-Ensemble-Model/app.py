import os
import io
import base64
import json
import re
import asyncio
import numpy as np
import cv2
import PIL.Image
from PIL import Image
from io import BytesIO
from typing import List, Optional, Dict
from scipy import stats

from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gradio as gr

import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import CLIPProcessor, CLIPModel
import google.generativeai as genai

# Google Cloud Vision imports with error handling
GCP_VISION_AVAILABLE = False
GCP_VISION_CLIENT = None
GCP_VISION_ERROR = "Not initialized"
try:
    from google.cloud import vision
    from google.oauth2 import service_account
    GCP_VISION_AVAILABLE = True
    print("✅ Google Cloud Vision imports successful")
except ImportError as e:
    GCP_VISION_AVAILABLE = False
    GCP_VISION_ERROR = f"Import failed: {str(e)}"
    print(f"❌ Google Cloud Vision not available: {e}")

print("===== Application Startup =====")

app = FastAPI(title="Upgraded MTurk Dual Ensemble + Gemini AI + GCP Vision Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== DUAL ENSEMBLE MODEL SETUP (Unchanged) =====
device = "cpu"
try:
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
    print("CLIP loaded!")
except Exception as e:
    print(f"Warning: CLIP load failed: {e}")

class Siamese(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(2048, 512)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

siamese = Siamese()
try:
    siamese.load_state_dict(torch.load("siamese_mturk.pth", map_location=device), strict=False)
    print("Siamese ResNet loaded from siamese_mturk.pth!")
except Exception as e:
    print(f"Warning: Could not load Siamese weights from siamese_mturk.pth: {e}")
    print("Using randomly initialized Siamese model")

siamese.eval().to(device)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# UPGRADE: Enhanced OpenCV for style (handles multi-req: avg feats for ref style)
def opencv_style_features(img_pil):
    try:
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        edges = cv2.Canny(gray, 50, 150)
        edge_mean = np.mean(edges)
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0]
        contrast = np.std(l_channel)
        brightness = np.mean(l_channel)
        hist = cv2.calcHist([img_cv], [0,1,2], None, [8,8,8], [0,256]*3).flatten()
        hist = hist / (hist.sum() + 1e-6)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        motion_var = np.var(sobelx + sobely)
        return np.concatenate([hist, [lap_var, edge_mean, contrast, brightness, motion_var]])
    except Exception as e:
        print(f"Error in opencv_style_features: {e}")
        return np.zeros(512 + 5)

# ===== NEW: Source Image Splitting Function =====
def split_source_image(source_image):
    """
    Split a source image into two halves for style annotation.
    Returns left and right halves as PIL Images.
    """
    try:
        if source_image is None:
            return None, None
        
        # Convert to PIL if it's a numpy array
        if isinstance(source_image, np.ndarray):
            source_image = Image.fromarray(source_image)
        
        width, height = source_image.size
        
        # Calculate split points - split horizontally
        split_point = width // 2
        
        # Left half
        left_half = source_image.crop((0, 0, split_point, height))
        
        # Right half  
        right_half = source_image.crop((split_point, 0, width, height))
        
        print(f"✅ Source image split successfully: {width}x{height} -> {split_point}x{height} each")
        
        return left_half, right_half
        
    except Exception as e:
        print(f"❌ Error splitting source image: {e}")
        return None, None

# ===== ENHANCED ANALYTICAL PATTERN-BASED IMAGE SELECTION (Unchanged) =====
def analyze_confidence_patterns(scores: Dict[str, List[float]]) -> Dict:
    custom = scores['custom']
    clip = scores['clip']
    resnet = scores['resnet']
    opencv_scores = scores['opencv']
    diff_scores = scores.get('diff', [0.5] * len(custom))
    
    num_images = len(custom)
    best_custom = max(range(num_images), key=lambda i: custom[i])
    best_clip = max(range(num_images), key=lambda i: clip[i])
    best_resnet = max(range(num_images), key=lambda i: resnet[i])
    best_opencv = max(range(num_images), key=lambda i: opencv_scores[i])
    
    resnet_range = max(resnet) - min(resnet)
    if resnet_range > 0.3:
        resnet_winner = best_resnet
        other_models_agree = sum([custom[resnet_winner] > 0.6, clip[resnet_winner] > 0.5])
        if other_models_agree >= 1:
            return {
                "selected_image": resnet_winner,
                "confidence": resnet[resnet_winner],
                "pattern": "resnet_clear_preference",
                "reasoning": f"ResNet clear pref for {resnet_winner+1} (range: {resnet_range:.3f}), {other_models_agree} supports"
            }

    def coefficient_of_variance(arr):
        return np.std(arr) / (np.mean(arr) + 1e-8)
    
    resnet_cv = coefficient_of_variance(resnet)
    clip_cv = coefficient_of_variance(clip)
    
    if resnet_cv < clip_cv and resnet_cv < 0.5:
        return {
            "selected_image": best_resnet,
            "confidence": resnet[best_resnet],
            "pattern": "resnet_lower_variance",
            "reasoning": f"ResNet consistent (CV: {resnet_cv:.3f} < CLIP {clip_cv:.3f})"
        }
    
    model_agreements = []
    for i in range(num_images):
        prefs = sum([custom[i] == max(custom), resnet[i] == max(resnet), clip[i] == max(clip), opencv_scores[i] == max(opencv_scores)])
        model_agreements.append(prefs)
    
    if best_custom == best_resnet and model_agreements[best_custom] >= 2:
        return {
            "selected_image": best_custom,
            "confidence": (custom[best_custom] + resnet[best_custom]) / 2,
            "pattern": "resnet_custom_agreement",
            "reasoning": f"ResNet/Custom agree on {best_custom+1} ({model_agreements[best_custom]}/4)"
        }
    
    resnet_z_scores = stats.zscore(resnet)
    significant_preferences = [i for i, z in enumerate(resnet_z_scores) if z > 1.0]
    if len(significant_preferences) == 1:
        sig_img = significant_preferences[0]
        return {
            "selected_image": sig_img,
            "confidence": resnet[sig_img],
            "pattern": "resnet_statistical_significance",
            "reasoning": f"ResNet sig pref {sig_img+1} (z: {resnet_z_scores[sig_img]:.2f})"
        }
    
    if best_custom == best_clip == best_resnet:
        avg = (custom[best_custom] + clip[best_custom] + resnet[best_custom]) / 3
        return {"selected_image": best_custom, "confidence": avg, "pattern": "all_agree", "reasoning": f"All agree on {best_custom+1}"}
    
    if best_custom == best_clip or best_custom == best_resnet:
        winner = best_custom
        avg = (custom[winner] + clip[winner] + resnet[winner]) / 3
        supports = [m for m in ["CLIP", "ResNet"] if locals()[f"best_{m.lower()}"] == winner]
        return {"selected_image": winner, "confidence": avg, "pattern": "majority_vote", "reasoning": f"Custom + {', '.join(supports)} on {winner+1}"}
    
    if best_clip == best_resnet:
        winner = best_clip
        avg = (custom[winner] + clip[winner] + resnet[winner]) / 3
        return {"selected_image": winner, "confidence": avg, "pattern": "majority_vote", "reasoning": f"CLIP+ResNet on {winner+1}"}
    
    ensemble_scores = []
    for i in range(num_images):
        score = (0.4 * resnet[i] + 0.35 * custom[i] + 0.15 * clip[i] + 0.1 * opencv_scores[i] + 0.15 * diff_scores[i])
        ensemble_scores.append(score)
    
    best = max(range(num_images), key=lambda i: ensemble_scores[i])
    details = [m for m, bs in [("ResNet", max(resnet)), ("Custom", max(custom)), ("CLIP", max(clip))] if locals()[f"best_{m.lower()}"] == best]
    detail_text = f" ({', '.join(details)})" if details else ""
    
    return {
        "selected_image": best,
        "confidence": ensemble_scores[best],
        "pattern": "analytical_weighted",
        "reasoning": f"Weighted favors {best+1}{detail_text} (score: {ensemble_scores[best]:.3f})"
    }

# ===== GOOGLE GEMINI SETUP (Unchanged) =====
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    print("⚠️ WARNING: No API key found! Set GOOGLE_API_KEY in HF Spaces secrets")

if API_KEY:
    genai.configure(api_key=API_KEY)

print("Google Gemini Vision API available")

GEMINI_MODELS = [
    "gemini-2.0-flash", # Best Refiner
    "gemini-3.0-pro", "gemini-3.0-flash", "gemini-3.0-deep-think",
    "gemini-2.5-pro", "gemini-2.5-flash",
    "gemini-1.5-pro", "gemini-1.5-flash"
]

MODEL_DESCRIPTIONS = {
    "gemini-3.0-pro": "🔥 New Pro - Most intelligent (50%+ reasoning)",
    "gemini-3.0-flash": "🚀 New Flash - Fast multimodal",
    "gemini-3.0-deep-think": "🧠 Preview Deep Think - Advanced reasoning",
    "gemini-2.5-pro": "⚖️ Stable Pro - High quality",
    "gemini-2.5-flash": "⚡ Stable Flash - Efficient",
    "gemini-2.0-flash": "🎯 Stable 2.0 - Reliable",
    "gemini-1.5-pro": "🛡️ Legacy Pro - Stable",
    "gemini-1.5-flash": "💨 Legacy Flash - Basic speed"
}

# ===== NEW PROMPT EVALUATION CRITERIA =====
PROMPT_EVAL_CRITERIA = """
Prompt Evaluation Criteria (order of importance):

1. EDITING INTENT:
   - Does the prompt clearly describe EDITING/COMBINING source images (not creating from scratch)?
   - Look for: "take", "replace", "add", "fix", "use", "remove", "keep", "combine"
   - Good: "Take X from image Y and replace Z" 
   - Bad: "Draw a new image of X"

2. ACCURACY & COMPLETENESS:
   - Does the prompt ACCURATELY describe the generated image (colors, style, objects, poses, background)?
   - NO extra elements that aren't in generated image
   - NO incorrect facts about generated image
   - Complete description of NEW elements/changes
   - Avoid unnecessary repetition of unchanged source details

3. CORRECT REFERENCES:
   - Are image number references ACCURATE? (e.g., "image 2" really contains mentioned object)
   - No mixed-up image numbers

4. INTERNAL CONSISTENCY:
   - No conflicting/contradictory instructions
   - No opposing statements (e.g., "replace background" vs "keep background")
   - No style conflicts

5. USE OF MULTIPLE SOURCES:
   - If result combines multiple sources, does prompt mention relevant ones?
   - Better prompts incorporate more relevant sources

6. GENERAL CORRECTNESS & CLARITY:
   - No meaningless decorative text ("masterpiece", "beautiful", etc.)
   - No irrelevant descriptions/storytelling
   - Correct grammar, spelling, syntax
   - No hallucinations about result image

Decision Process:
1. Check Editing Intent first
2. Verify Accuracy & Completeness  
3. Check Correct References & Internal Consistency
4. Consider Use of Multiple Sources
5. Final decision based on General Correctness

Output JSON: {"decision": "Prompt 1" or "Prompt 2" or "Same Quality", "confidence": 0.95, "reasoning": "Step-by-step evaluation per criteria"}
"""

class AnalyzeRequest(BaseModel):
    task_type: str
    prompt: Optional[str] = None
    images_b64: List[str]
    model_type: Optional[str] = "gemini"
    use_pattern_selection: Optional[bool] = True
    gemini_model: Optional[str] = "gemini"
    # New fields for prompt evaluation
    prompt1: Optional[str] = None
    prompt2: Optional[str] = None
    # NEW: Toggle for Enhanced/Dual Variation
    eval_mode: Optional[str] = "enhanced" 

def test_gemini_availability(model_name):
    if not API_KEY:
        return False, "API key not configured"
    try:
        available_models = genai.list_models()
        available_model_names = [model.name for model in available_models]
        model_found = any(model_name == model.split('/')[-1] for model in available_model_names)
        return model_found, f"Model {model_name} {'available' if model_found else 'not found'}"
    except Exception as e:
        return False, f"API error: {str(e)}"

def get_available_gemini_models():
    if not API_KEY:
        return []
    try:
        available_models = genai.list_models()
        available_model_names = [model.name for model in available_models]
        available_from_list = []
        for model in GEMINI_MODELS:
            for available_model in available_model_names:
                available_model_short = available_model.split('/')[-1]
                if model == available_model_short:
                    available_from_list.append(model)
                    print(f"✅ Found Gemini model: {model}")
                    break
        return available_from_list
    except Exception as e:
        print(f"❌ Error listing Gemini models: {e}")
        return []

AVAILABLE_GEMINI_MODELS = get_available_gemini_models()
DEFAULT_GEMINI_MODEL = AVAILABLE_GEMINI_MODELS[0] if AVAILABLE_GEMINI_MODELS else "gemini-2.0-flash"
DEFAULT_REFINER_MODEL = "gemini-2.0-flash" # Use fast model for refiner

print(f"Available Gemini models: {AVAILABLE_GEMINI_MODELS}")
print(f"Default Gemini model: {DEFAULT_GEMINI_MODEL}")

# ===== UPDATED MTURK CRITERIA PROMPT (With Realism Check) =====
CRITERIA_PROMPT = """
Selection Criteria (order of importance):
1. Prompt Compliance: Satisfy REFINED CHECKLIST exactly.
2. Realism & Natural Integration: The image must look realistic and physically plausible (unless an artistic style is explicitly requested). Edits must blend seamlessly with source lighting, grain, and shadows. Avoid "AI plastic/glossy" looks.
3. Anatomical/Scene Correctness: No errors (e.g., 6-leg horse, distorted hands). Correct perspective and physics.
4. Hallucinations: Did the model add objects not requested? (Strict Penalty).
5. Image Cohesion: The image must look like a single whole, not pasted elements.
6. Overall Quality: High resolution, sharpness, and correct exposure.
If tie, 'both'. Output criteria_scores [c1-6] (0-1 each).
"""

# ===== GOOGLE CLOUD VISION SETUP (Unchanged) =====
def setup_gcp_vision_client():
    global GCP_VISION_CLIENT, GCP_VISION_ERROR
    if not GCP_VISION_AVAILABLE:
        GCP_VISION_ERROR = "Google Cloud Vision library not installed"
        return None
    try:
        credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        print(f"🔍 Checking GCP credentials... Present: {bool(credentials_json)}")
        if credentials_json:
            credentials_info = json.loads(credentials_json)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            client = vision.ImageAnnotatorClient(credentials=credentials)
            GCP_VISION_ERROR = None
            print("✅ GCP Vision client initialized with JSON")
            return client
        else:
            try:
                client = vision.ImageAnnotatorClient()
                GCP_VISION_ERROR = None
                print("✅ GCP Vision client with default creds")
                return client
            except Exception as e:
                GCP_VISION_ERROR = f"Default creds failed: {e}"
                return None
    except Exception as e:
        GCP_VISION_ERROR = f"Setup error: {str(e)}"
        return None

GCP_VISION_CLIENT = setup_gcp_vision_client()
if not GCP_VISION_CLIENT:
    print(f"❌ GCP Vision failed: {GCP_VISION_ERROR}")
else:
    print("✅ GCP Vision ready!")

def clean_base64(b64_string: str) -> str:
    if "," in b64_string:
        return b64_string.split(",")[1]
    return b64_string

# ===== ALGORITHM UPGRADE: LLM-Based Prompt Refiner =====
def refine_prompt_with_gemini(raw_prompt: str, task_type: str = "image_eval") -> str:
    """
    Uses a fast Gemini model to convert messy user prompts into a strict evaluation checklist.
    This replaces regex/normalization for higher accuracy.
    """
    if not raw_prompt or len(raw_prompt) < 3:
        return "Criteria: Images must match reference inputs and general photography standards."
    
    try:
        # Use fast model for text processing if available, else default
        refiner_name = DEFAULT_REFINER_MODEL if DEFAULT_REFINER_MODEL in AVAILABLE_GEMINI_MODELS else DEFAULT_GEMINI_MODEL
        full_model_name = f"models/{refiner_name}"
        
        model = genai.GenerativeModel(full_model_name)
        
        system_instruction = f"""
        You are an expert AI Supervisor for MTurk Image Tasks. 
        Your goal is to convert a user's raw, potentially messy instruction into a strict PROMPT COMPLIANCE CHECKLIST.
        
        Task Type: {task_type}
        Raw Prompt: "{raw_prompt}"
        
        Rules:
        1. Extract the CORE INTENT (e.g., "Remove object", "Change color", "Fix blur").
        2. Identify CONSTRAINTS (e.g., "Keep background", "Don't change lighting").
        3. Identify SPECIFIC OBJECTS mentioned.
        4. Output a clean, numbered checklist.
        5. DO NOT add new requirements not implied by the prompt.
        
        Output format:
        PRIMARY GOAL: [Goal]
        CHECKLIST:
        1. [Requirement]
        2. [Requirement]
        NEGATIVE CONSTRAINTS:
        - [What must NOT happen]
        """
        
        response = model.generate_content(system_instruction)
        refined = response.text
        print(f"✨ REFINED PROMPT (Gemini):\n{refined}")
        return refined

    except Exception as e:
        print(f"⚠️ Prompt Refiner Failed: {e}")
        return raw_prompt # Fallback

# ===== UPDATED: Gemini ENHANCED Evaluation (Refiner + CoT + Realism) =====
def analyze_with_gemini_enhanced_eval(task_type: str, prompt: str, images_b64: List[str], gemini_model: str = None, diff_images_b64: List[str] = None):
    """
    Enhanced analysis: Refined Checklist -> Chain of Thought -> Decision
    """
    try:
        model_to_use = gemini_model or DEFAULT_GEMINI_MODEL
        if not model_to_use:
            raise HTTPException(status_code=503, detail="No Gemini model")
        
        full_model_name = f"models/{model_to_use}"
        print(f"Using Gemini for ENHANCED strict eval: {full_model_name}")
        model = genai.GenerativeModel(full_model_name)

        # STEP 1: Refine Prompt
        refined_criteria = refine_prompt_with_gemini(prompt, task_type)

        # Prepare PIL images
        response_images = []
        for b64 in images_b64:
            image_data = base64.b64decode(clean_base64(b64))
            response_images.append(PIL.Image.open(io.BytesIO(image_data)))

        diff_images = []
        if diff_images_b64:
            for b64 in diff_images_b64:
                image_data = base64.b64decode(clean_base64(b64))
                diff_images.append(PIL.Image.open(io.BytesIO(image_data)))
        
        # UPDATED Enhanced Prompt with Chain-of-Thought & Realism
        enhanced_prompt = f"""
ENHANCED IMAGE EVALUATION TASK (CHAIN-OF-THOUGHT MODE).

USER ORIGINAL INSTRUCTION: "{prompt}"

---
EVALUATION CHECKLIST (GROUND TRUTH):
{refined_criteria}
---

INSTRUCTIONS FOR VISION MODEL:
1. Review "Response Image 1" and "Response Image 2" against the CHECKLIST above.
2. If "Difference Images" are provided, use them to verify that ONLY the requested areas were changed.
3. **Realism Check:** Do the edits look like a real photo? (Lighting, shadows, grain match).
4. **Hallucination Check:** Did the model add random objects? (Reject if yes).

CRITERIA:
{CRITERIA_PROMPT}

OUTPUT FORMAT (STRICT JSON):
You MUST provide "step_by_step_analysis" FIRST to ensure reasoning precedes decision.

{{
  "step_by_step_analysis": "1. Checking Image 1 against checklist item 1... 2. Checking Image 2... 3. Comparing Realism...",
  "decision": 1 or 2 or "both",
  "scores": [score1, score2],
  "criteria_scores": [[c1..c6], [c1..c6]],
  "confidence": 0.95,
  "reasoning": "Final summary of why one is better."
}}"""

        # Build Content List (Interleaved Text & Images)
        content = []
        
        # Add Response Images (First 2 in list)
        for i, img in enumerate(response_images[:2]):
            content.append(f"Response Image {i+1}:")
            content.append(img)
            
        # Add Difference Images
        if diff_images:
            for i, img in enumerate(diff_images[:2]):
                content.append(f"Difference Image {i+1} (shows changes):")
                content.append(img)
                
        # Add Requested References (Explicitly labeled)
        if len(response_images) > 2:
            for i, img in enumerate(response_images[2:]):
                content.append(f"Requested Input Reference Image {i+1} (Input Image {i+1}):")
                content.append(img)
        
        # Add the Prompt Text at the end
        content.append(enhanced_prompt)

        try:
            response = model.generate_content(content)
            response_text = response.text
            print(f"Gemini Enhanced Eval response: {response_text[:200]}...")
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result["gemini_model_used"] = model_to_use
                result["model_description"] = MODEL_DESCRIPTIONS.get(model_to_use, "Gemini")
                result["eval_type"] = "enhanced_cot_refiner"
                result["refined_prompt"] = refined_criteria
                result["confidence"] = result.get("confidence", 0.5)
                return result
            else:
                 raise ValueError("Could not parse JSON from Gemini response")

        except Exception as e:
            print(f"Gemini Enhanced Eval Error: {e}")
            return analyze_with_gemini_direct_eval(task_type, prompt, images_b64, gemini_model, diff_images_b64)

    except Exception as e:
        print(f"Gemini Enhanced Setup Error: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini setup failed: {str(e)}")

# ===== GEMINI DIRECT EVALUATION (Standard / Old Dual Replacement) =====
def analyze_with_gemini_direct_eval(task_type: str, prompt: str, images_b64: List[str], gemini_model: str = None, diff_images_b64: List[str] = None):
    """
    Standard evaluation using Gemini.
    """
    try:
        model_to_use = gemini_model or DEFAULT_GEMINI_MODEL
        if not model_to_use:
            raise HTTPException(status_code=503, detail="No Gemini model")
        
        full_model_name = f"models/{model_to_use}"
        print(f"Using Gemini for direct eval: {full_model_name}")
        model = genai.GenerativeModel(full_model_name)
        
        images = []
        for b64 in images_b64:
            image_data = base64.b64decode(clean_base64(b64))
            img = PIL.Image.open(io.BytesIO(image_data))
            images.append(img)
        
        diff_images = []
        if diff_images_b64:
            for b64 in diff_images_b64:
                image_data = base64.b64decode(clean_base64(b64))
                img = PIL.Image.open(io.BytesIO(image_data))
                diff_images.append(img)
        
        eval_prompt = f"""
TASK: Evaluate how accurately the response images follow the user's instructions based on the provided criteria.

USER INSTRUCTIONS: "{prompt}"

{CRITERIA_PROMPT}

INSTRUCTIONS FOR AI:
1. Focus heavily on "Prompt Compliance". Did the image do exactly what was asked?
2. Use difference images (if provided) to verify that only requested changes occurred.
3. Compare response images against requested reference images (if provided) for context. 
   **NOTE:** Images labeled below as "Requested Input Reference Image" correspond to "Input Image 1", "Requested Image 1" in the instructions.
4. Pick the best image (1, 2, or "both" if tied).
5. Provide scores and detailed reasoning rooted in the criteria.
"""
        content = [eval_prompt]

        # Add Response Images (Image 1 & 2)
        for i, img in enumerate(images[:2]):
            content.append(f"Response Image {i+1}:")
            content.append(img)
        
        # Add Difference Images if available
        if diff_images:
            content.append("Difference Images (showing changes between responses and source):")
            for i, diff in enumerate(diff_images[:2]):
                content.append(f"Difference Image {i+1}:")
                content.append(diff)
        
        # UPDATED: Explicitly label Requested Reference Images
        if len(images) > 2:
            content.append("Requested Reference Images (Context/Source):")
            for i, img in enumerate(images[2:]):
                content.append(f"Requested Input Reference Image {i+1} (matches 'Input Image {i+1}' in prompt):")
                content.append(img)
        
        json_format = '{{"decision": 1 or 2 or "both", "scores": [s1,s2], "criteria_scores": [[c1_1..c5_1],[c1_2..c5_2]], "confidence": 0.95, "reasoning": "Step-by-step evaluation against criteria, focusing on prompt compliance."}}'
        content.append(f"Respond ONLY JSON: {json_format}")
        
        try:
            response = model.generate_content(content)
            response_text = response.text
            print(f"Gemini Direct Eval response: {response_text}")
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result["gemini_model_used"] = model_to_use
                result["model_description"] = MODEL_DESCRIPTIONS.get(model_to_use, "Gemini")
                result["eval_type"] = "direct_instruction_compliance"
                result["confidence"] = result.get("confidence", 0.5)
                return result
            else:
                 raise ValueError("Could not parse JSON from Gemini response")

        except Exception as e:
            print(f"Gemini Direct Eval Error: {e}")
            return {
                "decision": 1,
                "scores": [0.5, 0.5],
                "confidence": 0.1,
                "reasoning": f"Gemini evaluation failed: {str(e)[:100]}",
                "gemini_model_used": model_to_use,
                 "eval_type": "direct_eval_error"
            }

    except Exception as e:
        print(f"Gemini Eval Setup Error: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini setup failed: {str(e)}")

# ===== CORRECTED: Gemini Style Analysis WITHOUT Reference Images =====
def analyze_with_gemini_style(task_type: str, prompt: str, images_b64: List[str], gemini_model: str = None):
    """
    CORRECTED Style annotation: Uses Refiner -> Compares response images directly against style instructions.
    Does NOT use reference images.
    Returns decisions in "Hell Yes|Yes|No|Hell No" format with 13 criteria scores.
    """
    try:
        if task_type != "style_annot":
            raise ValueError("Style annotation function called for non-style task")
        
        # Ensure we only have the two response images, ignoring any extra references passed
        start_images_b64 = images_b64[:2]
        if len(start_images_b64) < 2:
             raise ValueError("Style annotation requires exactly 2 response images.")

        model_to_use = gemini_model or DEFAULT_GEMINI_MODEL
        if not model_to_use or model_to_use not in AVAILABLE_GEMINI_MODELS:
            return {
                "decision": "No", 
                "similarity": 0.5,
                "criteria_scores": [[0.5]*13, [0.5]*13],
                "reasoning": "No Gemini model available for style analysis",
                "model_used": "gemini_style_fallback"
            }
        
        full_model_name = f"models/{model_to_use}"
        print(f"Using Gemini for style analysis (no refs): {full_model_name}")
        model = genai.GenerativeModel(full_model_name)
        
        # STEP 1: Refine Style Guidelines
        refined_style = refine_prompt_with_gemini(prompt, "style_annotation")
        
        # Updated style prompt to NOT mention reference images
        style_prompt = f"""
STYLE ANNOTATION TASK:

Custom Task: Determine if response photos match the style guidelines provided in the USER INSTRUCTIONS.
Compare response images (1&2) directly against the required style in these aspects (score each 0-1):

---
REFINED STYLE CHECKLIST:
{refined_style}
---

1. Textures (grains, blur, noise, etc)
2. Shadows (soft, hard, etc)
3. Lighting (e.g., hard flash, natural light, etc)
4. Color temperature (e.g., warm, cold)
5. Color scheme (e.g., monochrome, grayscale, etc)
6. Brightness (e.g., dark low light vs vivid colorful)
7. Contrast
8. Light glare, bokeh effect
9. Motion rendering
10. Long exposure
11. Lens types/angle
12. Objects/background focus
13. Unique style features

Soft Labels: First decide hell_yes/hell_no (all perfect? / 1+ striking diff from requirement?), then yes/no.

USER INSTRUCTIONS (Style Guideline): "{prompt}"

Output JSON: {{"decision": "Hell Yes|Yes|No|Hell No", "similarity": avg 0-1, "criteria_scores": [[13 scores for resp1], [13 for resp2]], "reasoning": "Per criteria & which resp best matches the instructions"}}
"""
        
        # Prepare only the first two images (responses)
        images = []
        for b64 in start_images_b64:
            image_data = base64.b64decode(clean_base64(b64))
            img = PIL.Image.open(io.BytesIO(image_data))
            images.append(img)
        
        content = [style_prompt]
        
        # Add response images
        content.append("Response Image 1:")
        content.append(images[0])
        content.append("Response Image 2:")
        content.append(images[1])
        
        # NO requested references added here.
        
        response = model.generate_content(content)
        response_text = response.text
        print(f"Gemini style response (no refs): {response_text}")
        
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            result["model_used"] = "gemini_style_corrected_no_refs"
            result["gemini_model_used"] = model_to_use
            result["analysis_type"] = "style_annotation_13_criteria_direct"
            return result
        else:
            # Fallback with proper style annotation format
            return {
                "decision": "No",
                "similarity": 0.5,
                "criteria_scores": [[0.5]*13, [0.5]*13],
                "reasoning": f"Gemini parse fallback - analyzing style match for: {prompt}",
                "model_used": "gemini_style_fallback",
                "analysis_type": "style_annotation_13_criteria_direct"
            }
            
    except Exception as e:
        print(f"Gemini Style Error: {e}")
        return {
            "decision": "No",
            "similarity": 0.3,
            "criteria_scores": [[0.3]*13, [0.3]*13],
            "reasoning": f"Error: {str(e)[:100]} - style analysis failed",
            "model_used": "gemini_style_error",
            "analysis_type": "style_annotation_13_criteria_direct"
        }

# ===== NEW PROMPT EVALUATION FUNCTION (UPDATED with auto image role text) =====
def analyze_with_gemini_prompt_eval(source_images_b64: List[str], generated_image_b64: str, 
                                   prompt1: str, prompt2: str, gemini_model: str = None):
    """
    New function for prompt evaluation task
    Compares two prompts to see which better describes the editing process from sources to generated image
    """
    try:
        model_to_use = gemini_model or DEFAULT_GEMINI_MODEL
        if not model_to_use or model_to_use not in AVAILABLE_GEMINI_MODELS:
            return {
                "decision": "Same Quality", 
                "confidence": 0.5, 
                "reasoning": "No Gemini model available",
                "model_used": "gemini_prompt_eval_fallback"
            }
        
        full_model_name = f"models/{model_to_use}"
        print(f"Using Gemini for prompt evaluation: {full_model_name}")
        model = genai.GenerativeModel(full_model_name)
        
        # Prepare images
        content = ["PROMPT EVALUATION TASK: Determine which prompt better describes the editing process from SOURCE IMAGES to GENERATED IMAGE."]
        
        # Add source images
        content.append("SOURCE IMAGES (up to 10):")
        for i, b64 in enumerate(source_images_b64):
            image_data = base64.b64decode(clean_base64(b64))
            img = PIL.Image.open(io.BytesIO(image_data))
            content.append(f"Source Image {i+1}:")
            content.append(img)
        
        # Add generated image
        content.append("GENERATED IMAGE (result of editing):")
        image_data = base64.b64decode(clean_base64(generated_image_b64))
        img = PIL.Image.open(io.BytesIO(image_data))
        content.append(img)
        
        # Auto-add image role text and pick instruction
        auto_prompt_addition = f" The source images are Image 1-{len(source_images_b64)} and the generated image is Image {len(source_images_b64)+1}."
        
        # Add prompts and criteria
        content.append(f"""
PROMPTS TO COMPARE:
PROMPT 1: "{prompt1}"
PROMPT 2: "{prompt2}"
{auto_prompt_addition}

{PROMPT_EVAL_CRITERIA}

IMPORTANT: 
- Choose "Same Quality" ONLY if truly impossible to decide (both equally good/bad)
- Double-check before using "Same Quality"
- Consider number and importance of errors
""")
        
        json_format = '{"decision": "Prompt 1" or "Prompt 2" or "Same Quality", "confidence": 0.95, "reasoning": "Step-by-step evaluation"}'
        content.append(f"Respond ONLY JSON: {json_format}")
        
        response = model.generate_content(content)
        response_text = response.text
        print(f"Gemini prompt evaluation response: {response_text}")
        
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            result["model_used"] = "gemini_prompt_eval"
            result["gemini_model_used"] = model_to_use
            return result
        else:
            return {
                "decision": "Same Quality",
                "confidence": 0.5,
                "reasoning": "Failed to parse Gemini response - manual review needed",
                "model_used": "gemini_prompt_eval_parse_fallback"
            }
            
    except Exception as e:
        print(f"Gemini Prompt Evaluation Error: {e}")
        return {
            "decision": "Same Quality",
            "confidence": 0.3,
            "reasoning": f"Evaluation error: {str(e)[:100]}",
            "model_used": "gemini_prompt_eval_error"
        }

# ===== NEW: Function to handle source image splitting =====
def handle_source_image_split(source_image):
    """Split source image and update response image components"""
    if source_image is None:
        return [None, None, "⚠️ Please upload a source image first"]
    
    try:
        left_half, right_half = split_source_image(source_image)
        
        if left_half is None or right_half is None:
            return [None, None, "❌ Failed to split source image"]
        
        # Update the response image components
        return [
            left_half,  # image1
            right_half, # image2
            "✅ Source image split successfully! Response images updated automatically."
        ]
        
    except Exception as e:
        print(f"Error in handle_source_image_split: {e}")
        return [None, None, f"❌ Error splitting image: {str(e)}"]

# ===== COMMON FUNCTIONS =====
def decode_base64_image(b64_string: str) -> Image.Image:
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        img_data = base64.b64decode(b64_string)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Invalid image: {str(e)}")

# ===== FASTAPI ENDPOINTS =====
@app.get("/")
async def root():
    gemini_available = len(AVAILABLE_GEMINI_MODELS) > 0
    gcp_vision_available = GCP_VISION_CLIENT is not None
    return {
        "status": "online",
        "message": "MTurk Dual Ensemble + Gemini AI + GCP Vision Assistant - CORRECTED Style Annotation",
        "models": {
            "dual_ensemble": "active",
            "gemini_vision": "available" if gemini_available else "unavailable",
            "gcp_vision": "available" if gcp_vision_available else "unavailable",
            "analytical_pattern_selection": "active"
        },
        "available_gemini_models": AVAILABLE_GEMINI_MODELS,
        "default_gemini_model": DEFAULT_GEMINI_MODEL,
        "gcp_vision_available": gcp_vision_available,
        "gcp_vision_error": GCP_VISION_ERROR if not gcp_vision_available else None,
        "features": {
            "style_annotation_corrected": "Active (Hell Yes|Yes|No|Hell No decisions, NO references)",
            "source_image_splitting": "Active for style annotation", 
            "13_criteria_analysis": "Active for style annotation",
            "prompt_evaluation": "Active (MTurk 6-step criteria)",
            "image_evaluation": "Active (Enhanced & Direct modes available)"
        },
        "endpoints": {"/analyze": "POST Form", "/api/analyze": "POST JSON", "/health": "GET"}
    }

@app.get("/health")
async def health():
    gcp_vision_available = GCP_VISION_CLIENT is not None
    gemini_status = {}
    for model in AVAILABLE_GEMINI_MODELS:
        available, message = test_gemini_availability(model)
        gemini_status[model] = {
            "available": available,
            "message": message,
            "description": MODEL_DESCRIPTIONS.get(model, "Gemini Model")
        }
    
    return {
        "status": "healthy" if (AVAILABLE_GEMINI_MODELS or GCP_VISION_CLIENT) else "degraded",
        "models_loaded": True,
        "device": device,
        "gemini_available": len(AVAILABLE_GEMINI_MODELS) > 0,
        "gemini_models": gemini_status,
        "default_gemini_model": DEFAULT_GEMINI_MODEL,
        "gcp_vision_available": gcp_vision_available,
        "gcp_vision_error": GCP_VISION_ERROR if not gcp_vision_available else None,
        "style_annotation_corrections": {
            "decision_format": "Hell Yes|Yes|No|Hell No (CORRECTED)",
            "criteria_count": "13 criteria (CORRECTED)",
            "reference_images": "DISABLED for style annotation",
            "soft_labels": "Active - hell_yes/hell_no then yes/no",
            "similarity_score": "Average 0-1 across criteria"
        },
        "active_task_types": ["image_eval", "style_annot", "prompt_eval"]
    }

@app.post("/analyze")
async def analyze(images_b64: str = Form(...)):
    # Placeholder for legacy ensemble endpoint if needed, though not heavily used in this context.
    return {"error": "Use /api/analyze endpoint for current functionality"}

@app.post("/api/analyze")
async def analyze_api_endpoint(request: AnalyzeRequest):
    return await analyze_logic(request)

async def analyze_logic(request: AnalyzeRequest):
    try:
        images = []
        # Decode images first to check validity and count
        for i, b64 in enumerate(request.images_b64):
            try:
                img = decode_base64_image(b64)
                images.append(img)
            except Exception as e:
                print(f"Decode warning {i}: {e}")
                
        if len(images) == 0:
            raise HTTPException(status_code=400, detail="No valid images")
            
        # Handle different task types
        if request.task_type == "prompt_eval":
            if not API_KEY:
                raise HTTPException(status_code=503, detail="Gemini key missing for prompt evaluation")
            if len(AVAILABLE_GEMINI_MODELS) == 0:
                raise HTTPException(status_code=503, detail="No Gemini models available")
                
            if len(images) < 2:
                raise HTTPException(status_code=400, detail="Prompt evaluation requires at least 1 source and 1 generated image")
                
            if not request.prompt1 or not request.prompt2:
                raise HTTPException(status_code=400, detail="Prompt evaluation requires both prompt1 and prompt2")
                
            # Re-encode for Gemini function
            images_b64_for_gemini = []
            for img in images:
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                images_b64_for_gemini.append(f"data:image/png;base64,{img_str}")
                
            generated_image_b64 = images_b64_for_gemini[0]
            source_images_b64 = images_b64_for_gemini[1:]
            
            gemini_model_to_use = request.gemini_model or DEFAULT_GEMINI_MODEL
            
            result = analyze_with_gemini_prompt_eval(
                source_images_b64, generated_image_b64, 
                request.prompt1, request.prompt2, gemini_model_to_use
            )
            result["model_used"] = "gemini_prompt_eval"
            
        elif request.task_type == "style_annot":
            # CORRECTED: Use proper style annotation with Hell Yes|Yes|No|Hell No decisions & NO REFERENCES
            if not API_KEY:
                raise HTTPException(status_code=503, detail="Gemini key missing for style analysis")
            if len(AVAILABLE_GEMINI_MODELS) == 0:
                raise HTTPException(status_code=503, detail="No Gemini models available")
            
            if len(images) < 2:
                 raise HTTPException(status_code=400, detail="Style annotation requires exactly 2 response images")

            # Re-encode only the first two images for Gemini function
            images_b64_for_gemini = []
            for img in images[:2]:
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                images_b64_for_gemini.append(f"data:image/png;base64,{img_str}")
                
            gemini_model_to_use = request.gemini_model or DEFAULT_GEMINI_MODEL
            
            result = analyze_with_gemini_style(
                request.task_type, request.prompt, images_b64_for_gemini, gemini_model_to_use
            )
            result["model_used"] = "gemini_style_corrected_no_refs"
            
        elif request.model_type == "gemini":
            # This block handles image_eval or generic gemini requests
            if not API_KEY:
                raise HTTPException(status_code=503, detail="Gemini key missing")
            if len(AVAILABLE_GEMINI_MODELS) == 0:
                raise HTTPException(status_code=503, detail="No Gemini models")
                
            images_b64_for_gemini = []
            for img in images:
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                images_b64_for_gemini.append(f"data:image/png;base64,{img_str}")
                
            gemini_model_to_use = request.gemini_model or DEFAULT_GEMINI_MODEL
            
            # Assume images 2 and 3 (indices 2,3) are diffs if present based on Gradio logic
            diff_b64 = images_b64_for_gemini[2:4] if len(images_b64_for_gemini) >= 4 else []
            # Remaining images are references (indices 4 onwards)
            # Combine responses (0,1) and references (4+) for main image list
            refs_b64 = images_b64_for_gemini[4:] if len(images_b64_for_gemini) > 4 else []
            main_images_b64 = images_b64_for_gemini[:2] + refs_b64

            # CHECK EVAL MODE TO SELECT LOGIC
            if request.task_type == "image_eval" and request.eval_mode == "enhanced":
                 # Use the new ENHANCED logic
                 result = analyze_with_gemini_enhanced_eval(
                    request.task_type, request.prompt or "evaluate these images",
                    main_images_b64, gemini_model_to_use, diff_b64
                )
                 result["model_used"] = "gemini_vision_enhanced_strict"
            else:
                # Use Standard/Direct Evaluation
                result = analyze_with_gemini_direct_eval(
                    request.task_type, request.prompt or "evaluate these images",
                    main_images_b64, gemini_model_to_use, diff_b64
                )
                result["model_used"] = "gemini_direct_eval"
                
        elif request.model_type == "gcp_vision":
             # Placeholder for GCP logic if needed in future
             raise HTTPException(status_code=501, detail="GCP Vision path not implemented in this update.")
        else:
             # Placeholder for Ensemble logic if needed in future
             raise HTTPException(status_code=501, detail="Ensemble path not implemented in this update.")
            
        result["task_type"] = request.task_type
        if request.task_type in ["image_eval", "style_annot"] and request.prompt:
            result["original_prompt"] = request.prompt
            
        if "model_used" not in result:
            result["model_used"] = "unknown"
            
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Analyze error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== GRADIO UI WITH CORRECTED STYLE ANNOTATION & EVAL =====
def gradio_analyze(model_type, task_type, prompt, image1, image2, diff_image1, diff_image2, 
                  prompt1, prompt2, eval_mode, style_source_image, *args):
    extra_images_list = list(args[:8])
    extra_count = args[8]
    use_pattern = args[9]
    gemini_model = args[10]
    
    images_b64 = []
    
    # Handle different task types
    if task_type == "prompt_eval":
        if image1 is None:
            return {"error": "Need Generated Image for prompt evaluation"}
        # For prompt eval: image1 is generated, extra images are sources
        required_images = [image1]
    else:
        if image1 is None or image2 is None:
            return {"error": "Need Response Image 1 & 2"}
        required_images = [image1, image2]
    
    # Add required images (Responses or Generated)
    for img in required_images:
        if img is not None:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            images_b64.append(f"data:image/png;base64,{img_str}")
    
    # Diffs only for image_eval
    if task_type == "image_eval":
        diff_images = [diff_image1, diff_image2]
        for img in diff_images:
            if img is not None:
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                images_b64.append(f"data:image/png;base64,{img_str}")
    
    # Requested/source images - used for image_eval (as context) and prompt_eval (as sources)
    # NOT used for style_annot anymore.
    if task_type in ["image_eval", "prompt_eval"]:
        requested = [img for img in extra_images_list[:extra_count] if img is not None]
        for img in requested:
            if img is not None:
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                images_b64.append(f"data:image/png;base64,{img_str}")
    
    images_b64 = [img for img in images_b64 if img is not None]
    
    request = AnalyzeRequest(
        model_type=model_type,
        task_type=task_type,
        prompt=prompt if prompt else None,
        images_b64=images_b64,
        use_pattern_selection=use_pattern,
        gemini_model=gemini_model if model_type == "gemini" else None,
        prompt1=prompt1 if task_type == "prompt_eval" else None,
        prompt2=prompt2 if task_type == "prompt_eval" else None,
        eval_mode=eval_mode # Pass the toggle value
    )
    
    result = asyncio.run(analyze_logic(request))
    return result

def get_gemini_model_choices():
    choices = []
    for model in GEMINI_MODELS:
        desc = MODEL_DESCRIPTIONS.get(model, "Gemini")
        is_avail = model in AVAILABLE_GEMINI_MODELS
        status = "✅" if is_avail else "❌"
        display = f"{status} {model} - {desc}"
        choices.append((display, model))
    return choices

# Helper functions for extra images
def add_extra(count):
    if count < 8:
        updates = [gr.update(visible=True) if i == count else gr.update() for i in range(8)]
        add_btn_visible = count < 7
        remove_btn_visible = count >= 0
        clear_btn_visible = count >= 0
        return updates + [gr.update(visible=add_btn_visible), gr.update(visible=remove_btn_visible), gr.update(visible=clear_btn_visible), count + 1]
    return [gr.update() for _ in range(8)] + [gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), count]

def remove_extra(count):
    if count > 0:
        new_count = count - 1
        updates = []
        for i in range(8):
            if i == count - 1:
                updates.append(gr.update(visible=False, value=None))
            elif i < count - 1:
                updates.append(gr.update(visible=True))
            else:
                updates.append(gr.update(visible=False))
        add_btn_visible = new_count < 8
        remove_btn_visible = new_count > 0
        clear_btn_visible = new_count > 0
        return updates + [gr.update(visible=add_btn_visible), gr.update(visible=remove_btn_visible), gr.update(visible=clear_btn_visible), new_count]
    return [gr.update(visible=False, value=None) for _ in range(8)] + [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), 0]

def clear_all_requested(count):
    updates = [gr.update(visible=False, value=None) for _ in range(8)]
    return updates + [gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), 0]

initial_extra_state = [None] * 8

with gr.Blocks(title="MTurk AI Assistant - UPDATED Style & Eval") as demo:
    gr.Markdown("# 🤖 MTurk AI Assistant -  MTurk Dual Ensemble + Gemini AI")
    gr.Markdown("**Features:** Style Annotation (No Refs) • Direct Instruction Compliance Eval • Source Splitting • Gemini AI")
    
    with gr.Row():
        model_type = gr.Dropdown(choices=["gemini"], value="gemini", label="AI Model (Gemini Only for these tasks)", interactive=False)
        task_type = gr.Dropdown(choices=["image_eval", "style_annot", "prompt_eval"], value="style_annot", label="Task Type")
        use_pattern = gr.Checkbox(value=False, label="Enable Pattern Selection", info="Not used currently", visible=False)
    
    gemini_model = gr.Dropdown(
        choices=get_gemini_model_choices(),
        value=DEFAULT_GEMINI_MODEL if AVAILABLE_GEMINI_MODELS else None,
        label="Gemini Model",
        visible=True,
        info="Flash/Pro variants"
    )

    # NEW: Evaluation mode toggle for image_eval
    eval_mode_row = gr.Row(visible=False)
    with eval_mode_row:
        eval_mode = gr.Radio(
            choices=["dual_variation", "enhanced"],
            value="enhanced",
            label="🎛️ Image Evaluation Mode",
            info="Enhanced = Strict Prompt Compliance (Recommended) | Dual Variation = Standard Direct Eval"
        )
    
    # Prompt inputs for prompt evaluation
    prompt_eval_row = gr.Row(visible=False)
    with prompt_eval_row:
        gr.Markdown("### 📝 Prompts to Compare (Prompt Evaluation Only)")
        with gr.Row():
            prompt1 = gr.Textbox(label="Prompt 1", placeholder="First editing prompt...", lines=2)
            prompt2 = gr.Textbox(label="Prompt 2", placeholder="Second editing prompt...", lines=2)
    
    prompt = gr.Textbox(
        label="User Instructions / Style Guidelines",
        placeholder="e.g., 'Make image vintage style' or 'remove the object'",
        info="Instructions the images will be evaluated against.",
        visible=True
    )
    
    # Source Image Section for Style Annotation
    style_source_row = gr.Row(visible=True)
    with style_source_row:
        gr.Markdown("### 🎪 Source Image (Style Annotation Splitter)")
        gr.Markdown("**Upload a source image containing 2 images side by side. Click 'Split Source Image' to automatically populate response images.**")
        with gr.Row():
            style_source_image = gr.Image(
                type="pil", 
                label="Source Image (to split)",
                sources=["upload", "clipboard"]
            )
            split_btn = gr.Button("✂️ Split Source Image", variant="secondary")
        split_status = gr.Textbox(
            label="Split Status",
            interactive=False,
            placeholder="Upload a source image and click 'Split Source Image'...",
            max_lines=2
        )
    
    gr.Markdown("### 🎯 Response Images (or Generated Image for Prompt Eval)")
    with gr.Row():
        image1 = gr.Image(type="pil", label="Response Image 1", sources=["upload", "clipboard"])
        image2 = gr.Image(type="pil", label="Response Image 2", sources=["upload", "clipboard"])
    
    diff_row = gr.Row(visible=False)
    with diff_row:
        gr.Markdown("### 🔄 Difference Images (Image Eval Only)")
        with gr.Row():
            diff_image1 = gr.Image(type="pil", label="Difference Image for Response 1", sources=["upload", "clipboard"])
            diff_image2 = gr.Image(type="pil", label="Difference Image for Response 2", sources=["upload", "clipboard"])
    
    # Requested images accordion - HIDDEN for style_annot now.
    req_acc = gr.Accordion("📁 Requested Reference/Source Images (For Image Eval & Prompt Eval)", open=False, visible=False)
    with req_acc:
        gr.Markdown("**Context Images:** References for evaluation context or sources for prompt evaluation.")
        extra_images = []
        extra_containers = []
        for i in range(8):
            with gr.Row(visible=False) as container:
                extra_img = gr.Image(type="pil", label=f"Reference/Source Image {i+1}", sources=["upload", "clipboard"])
                extra_images.append(extra_img)
                extra_containers.append(container)
        
        extra_count = gr.State(0)
        
        with gr.Row():
            add_btn = gr.Button("➕ Add Extra Image", visible=True)
            remove_btn = gr.Button("➖ Remove Last Extra", visible=False)
            clear_all_btn = gr.Button("🗑️ Clear All Extras", visible=False)
        
        add_btn.click(
            fn=add_extra,
            inputs=extra_count,
            outputs=extra_containers + [add_btn, remove_btn, clear_all_btn, extra_count]
        )
        remove_btn.click(
            fn=remove_extra,
            inputs=extra_count,
            outputs=extra_containers + [add_btn, remove_btn, clear_all_btn, extra_count]
        )
        clear_all_btn.click(
            fn=clear_all_requested,
            inputs=extra_count,
            outputs=extra_containers + [add_btn, remove_btn, clear_all_btn, extra_count]
        )
    
    analyze_btn = gr.Button("🔍 Analyze with AI", variant="primary")
    output = gr.JSON(label="Result")
    
    # Split button functionality
    split_btn.click(
        fn=handle_source_image_split,
        inputs=[style_source_image],
        outputs=[image1, image2, split_status]
    )
    
    def update_ui_for_task_type(task_type_val):
        if task_type_val == "prompt_eval":
            return [
                gr.update(visible=True),   # prompt_eval_row
                gr.update(visible=False),  # diff_row  
                gr.update(visible=True, open=True), # req_acc (Sources)
                gr.update(label="Generated Image"),  # image1
                gr.update(visible=False),  # image2
                gr.update(visible=False),  # prompt (regular)
                gr.update(visible=False),  # style_source_row
                gr.update(visible=False)   # eval_mode_row
            ]
        elif task_type_val == "image_eval":
            return [
                gr.update(visible=False),  # prompt_eval_row
                gr.update(visible=True),   # diff_row
                gr.update(visible=True, open=True), # req_acc (References)
                gr.update(label="Response Image 1"),  # image1
                gr.update(visible=True),   # image2
                gr.update(visible=True, label="User Instructions"), # prompt (regular)
                gr.update(visible=False),   # style_source_row
                gr.update(visible=True)    # eval_mode_row
            ]
        else:  # style_annot
            return [
                gr.update(visible=False),  # prompt_eval_row
                gr.update(visible=False),  # diff_row
                gr.update(visible=False, open=False), # req_acc - HIDDEN for style_annot
                gr.update(label="Response Image 1"),  # image1
                gr.update(visible=True),   # image2
                gr.update(visible=True, label="Style Guidelines"), # prompt (regular)
                gr.update(visible=True),    # style_source_row - VISIBLE for splitter
                gr.update(visible=False)    # eval_mode_row
            ]
    
    def update_gemini_vis(model_type_val):
        return gr.update(visible=(model_type_val == "gemini"))
    
    task_type.change(
        fn=update_ui_for_task_type, 
        inputs=task_type, 
        outputs=[prompt_eval_row, diff_row, req_acc, image1, image2, prompt, style_source_row, eval_mode_row]
    )
    model_type.change(fn=update_gemini_vis, inputs=model_type, outputs=gemini_model)
    
    analyze_btn.click(
        fn=gradio_analyze,
        inputs=[
            model_type, task_type, prompt,
            image1, image2, diff_image1, diff_image2,
            prompt1, prompt2, eval_mode, style_source_image,
            *extra_images,
            extra_count, use_pattern, gemini_model
        ],
        outputs=output
    )
    
    with gr.Accordion("🔧 Documentation & Updates", open=True):
        gr.Markdown(f"""
### 🆕 Latest Updates
        
        **1. Style Annotation (UPDATED):**
        - **No Reference Images:** Analysis now compares response images directly against the style guidelines provided in the prompt text.
        - **Direct Instruction Check:** The AI evaluates how well the images adhere to the written style criteria.
        - **13 Criteria:** Continues to use the detailed 13-point style checklist with "Hell Yes/No" decisions.
        
        **2. Image Evaluation (UPDATED):**
        - **Enhanced Mode (Strict Prompt Compliance):** New toggle allows you to use the "Enhanced" strict logic ported from Claude. This prioritizes exact adherence to user instructions and uses interleaved image/text prompts to better identify "Requested" images.
        - **Dual Variation/Direct:** Option to use the standard Direct Evaluation logic.
        - **Uses Context:** Enhanced mode explicitly maps "Requested Input Reference Image" labels to help Gemini understand the context better.

        ### Setup Instructions
        
        1. **Add API Key to HF Spaces:** Settings → Variables and Secrets → `GOOGLE_API_KEY`.
        
        **Current Status:**
        - Available Gemini Models: {len(AVAILABLE_GEMINI_MODELS)}
        - Default Model: `{DEFAULT_GEMINI_MODEL}`
        - API Status: {'✅ Ready' if API_KEY else '❌ No API Key'}
        """)
        
print("===== Application Ready =====")
print(f"Style Annotation: Active (Direct prompt comparison, NO references)")
print(f"Image Evaluation: Active (Enhanced & Direct modes available)")
print(f"Prompt Evaluation: Active (MTurk 6-step criteria)")
print(f"Available Gemini: {AVAILABLE_GEMINI_MODELS}")

demo.queue().launch()