#!/usr/bin/env python3
"""
Osteoarthritis Clinical Decision Support System - Standalone Version
Complete clinical application with AI-powered X-ray analysis and treatment planning.
"""

import os, sys, json, datetime, io, re
try:
    import tomllib as _toml
except Exception:
    _toml = None
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as tv_models

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# PDF generation imports
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Optional LLM support (supports both legacy and >=1.0 SDKs)
try:
    import openai  # module present in both SDKs
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False

try:
    from openai import OpenAI as OpenAIClient  # >=1.0
    OPENAI_V1 = True
except Exception:
    OPENAI_V1 = False

# === Configuration and Setup ===

def get_repo_root(marker_files=(".git", "pyproject.toml", "requirements.txt")) -> Path:
    """Walk upward until we hit a repo marker; fallback to '..'."""
    cur = Path.cwd().resolve()
    for parent in [cur, *cur.parents]:
        if any((parent / m).exists() for m in marker_files):
            return parent
    return Path("..").resolve()

# Repository configuration
REPO_ROOT = get_repo_root()
REPO_NAME = REPO_ROOT.name
sys.path.append(str(REPO_ROOT))

# Official Color Palette for consistent UI/visualizations
COLORS = {
    'primary': {'medical_blue': '#2E5BBA', 'healthcare_teal': '#1B998B', 'clinical_purple': '#6A4C93'},
    'neutral': {'charcoal': '#2C3E50', 'slate_gray': '#5D6D7E', 'light_gray': '#BDC3C7', 'off_white': '#F8F9FA'},
    'semantic': {'success_green': '#27AE60', 'warning_orange': '#E67E22', 'error_red': '#E74C3C', 'info_blue': '#3498DB'},
    'severity': {'Normal': '#2ECC71', 'Doubtful': '#F1C40F', 'Mild': '#E67E22', 'Moderate': '#E74C3C', 'Severe': '#8E44AD'},
    'models': {'efficientnet': '#2E5BBA', 'regnet': '#1B998B', 'densenet': '#6A4C93', 'resnet': '#27AE60', 'convnext': '#FF9A8B'}
}

def get_openai_key():
    """Get OpenAI API key from Streamlit secrets or environment.

    Supports any of these shapes in .streamlit/secrets.toml:
      OPENAI_API_KEY = "..."
      [api_keys]\nopenai_api_key = "..."
      [openai]\napi_key = "..."
    """
    try:
        if hasattr(st, 'secrets') and st.secrets:
            if 'OPENAI_API_KEY' in st.secrets:
                return st.secrets['OPENAI_API_KEY']
            if 'api_keys' in st.secrets and isinstance(st.secrets['api_keys'], dict):
                if 'openai_api_key' in st.secrets['api_keys']:
                    return st.secrets['api_keys']['openai_api_key']
            if 'openai' in st.secrets and isinstance(st.secrets['openai'], dict):
                if 'api_key' in st.secrets['openai']:
                    return st.secrets['openai']['api_key']
    except Exception:
        pass
    # Environment variable fallback
    env_key = os.getenv('OPENAI_API_KEY')
    if env_key:
        return env_key
    # Local file fallback: parse .streamlit/secrets.toml manually if Streamlit didn't load it
    try:
        if _toml:
            candidates = [
                REPO_ROOT / '.streamlit' / 'secrets.toml',
                Path.cwd() / '.streamlit' / 'secrets.toml',
            ]
            for p in candidates:
                if p.exists():
                    with open(p, 'rb') as fh:
                        data = _toml.load(fh)
                    if isinstance(data, dict):
                        if 'OPENAI_API_KEY' in data:
                            return data['OPENAI_API_KEY']
                        if 'api_keys' in data and isinstance(data['api_keys'], dict) and 'openai_api_key' in data['api_keys']:
                            return data['api_keys']['openai_api_key']
                        if 'openai' in data and isinstance(data['openai'], dict) and 'api_key' in data['openai']:
                            return data['openai']['api_key']
    except Exception:
        pass
    return None

def chat_completion(messages, max_tokens=1000, temperature=0.3, model: Optional[str] = None) -> str:
    """Unified chat completion for both OpenAI SDKs.

    - Uses >=1.0 client if available, else falls back to legacy ChatCompletion.
    - Raises on failure so callers can surface meaningful errors.
    """
    api_key = get_openai_key()
    if not api_key or not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI API key not configured or SDK unavailable")

    if OPENAI_V1:
        client = OpenAIClient(api_key=api_key)
        mdl = model or "gpt-4o-mini"
        resp = client.chat.completions.create(
            model=mdl,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content
    else:
        # Legacy SDK path (<=0.28)
        openai.api_key = api_key
        mdl = model or "gpt-4"
        resp = openai.ChatCompletion.create(
            model=mdl,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content

# === Model Classes ===

MODEL_BUILDERS = {
    "resnet50":        lambda cfg: _build_resnet50(cfg),
    "resnet":          lambda cfg: _build_resnet50(cfg),
    "densenet121":     lambda cfg: _build_densenet121(cfg),
    "densenet":        lambda cfg: _build_densenet121(cfg),
    "efficientnet_b0": lambda cfg: _build_efficientnet_b0(cfg),
    "efficientnet":    lambda cfg: _build_efficientnet_b0(cfg),
    "convnext_tiny":   lambda cfg: _build_convnext_tiny(cfg),
    "convnext":        lambda cfg: _build_convnext_tiny(cfg),
    "regnet_y_800mf":  lambda cfg: _build_regnet_y_800mf(cfg),
    "regnet":          lambda cfg: _build_regnet_y_800mf(cfg),
}

def _build_resnet50(cfg):
    m = tv_models.resnet50(weights=None)
    hidden_dim = cfg.get("hidden_dim", 384)
    num_classes = cfg.get("num_classes", 5)
    dropout_rate = cfg.get("dropout", 0.1)
    
    # Store original fc in_features before replacing
    in_features = m.fc.in_features
    
    # Build fc to match training exactly: [dropout, linear_hidden, relu, dropout, linear_output]
    m.fc = nn.Sequential(
        nn.Dropout(dropout_rate),                     # fc[0]
        nn.Linear(in_features, hidden_dim),           # fc[1]  
        nn.ReLU(),                                    # fc[2]
        nn.Dropout(dropout_rate),                     # fc[3]
        nn.Linear(hidden_dim, num_classes),           # fc[4]
    )
    return m

def _build_densenet121(cfg):
    m = tv_models.densenet121(weights=None)
    hidden_dim = cfg.get("hidden_dim", 512)
    num_classes = cfg.get("num_classes", 5)
    dropout_rate = cfg.get("dropout", 0.4)
    
    # Store original classifier in_features before replacing
    in_features = m.classifier.in_features
    
    # Build classifier to match training exactly: [dropout, linear_hidden, relu, dropout, linear_output]
    m.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),                     # classifier[0]
        nn.Linear(in_features, hidden_dim),           # classifier[1]
        nn.ReLU(),                                    # classifier[2]
        nn.Dropout(dropout_rate),                     # classifier[3]
        nn.Linear(hidden_dim, num_classes),           # classifier[4]
    )
    return m

def _build_efficientnet_b0(cfg):
    m = tv_models.efficientnet_b0(weights=None)
    hidden_dim = cfg.get("hidden_dim", 384)
    num_classes = cfg.get("num_classes", 5)
    dropout_rate = cfg.get("dropout", 0.4)
    
    # Store original classifier in_features before replacing
    in_features = m.classifier[1].in_features
    
    # Build classifier to match training exactly: [dropout, linear_hidden, relu, dropout, linear_output]
    m.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),                     # classifier[0]
        nn.Linear(in_features, hidden_dim),           # classifier[1] 
        nn.ReLU(),                                    # classifier[2]
        nn.Dropout(dropout_rate),                     # classifier[3]
        nn.Linear(hidden_dim, num_classes),           # classifier[4]
    )
    return m

def _build_convnext_tiny(cfg):
    m = tv_models.convnext_tiny(weights=None)
    hidden_dim = cfg.get("hidden_dim", 512)
    num_classes = cfg.get("num_classes", 5)
    dropout_rate = cfg.get("dropout", 0.2)
    
    # Store original classifier in_features before replacing last layer
    in_features = m.classifier[-1].in_features
    
    # Build classifier head to match training exactly: [dropout, linear_hidden, relu, dropout, linear_output]
    head = nn.Sequential(
        nn.Dropout(dropout_rate),                     # head[0]
        nn.Linear(in_features, hidden_dim),           # head[1]
        nn.ReLU(),                                    # head[2]
        nn.Dropout(dropout_rate),                     # head[3]
        nn.Linear(hidden_dim, num_classes),           # head[4]
    )
    
    # Replace only the last layer (like training code does)
    m.classifier[-1] = head
    return m

def _build_regnet_y_800mf(cfg):
    m = tv_models.regnet_y_800mf(weights=None)
    hidden_dim = cfg.get("hidden_dim", 768)
    num_classes = cfg.get("num_classes", 5)
    dropout_rate = cfg.get("dropout", 0.2)
    
    # Store original fc in_features before replacing
    in_features = m.fc.in_features
    
    # Build fc to match training exactly: [dropout, linear_hidden, relu, dropout, linear_output]
    m.fc = nn.Sequential(
        nn.Dropout(dropout_rate),                     # fc[0]
        nn.Linear(in_features, hidden_dim),           # fc[1]
        nn.ReLU(),                                    # fc[2]
        nn.Dropout(dropout_rate),                     # fc[3]
        nn.Linear(hidden_dim, num_classes),           # fc[4]
    )
    return m

class EnsembleModel(nn.Module):
    """Weighted-logit ensemble used in deployment."""
    def __init__(self, models: List[nn.Module], weights: np.ndarray, device: str):
        super().__init__()
        self.device = device
        self.models = nn.ModuleList([m.to(device) for m in models])
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32, device=device))

    def forward(self, x, return_logits: bool = True):
        x = x.to(self.device)
        logits_list = []
        for m in self.models:
            m.eval()
            with torch.no_grad():
                logits_list.append(m(x))
        stacked = torch.stack(logits_list, dim=0)
        w = self.weights.view(-1, 1, 1)
        ens_logits = (stacked * w).sum(dim=0)
        return ens_logits if return_logits else F.softmax(ens_logits, dim=1)

class TemperatureScaling(nn.Module):
    """Wrapper to apply learned temperature on logits."""
    def __init__(self, model: nn.Module, temperature: float, device: str):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.tensor([temperature], device=device), requires_grad=False)

    def forward(self, x, return_logits: bool = True):
        logits = self.model(x, return_logits=True)
        scaled = logits / self.temperature
        return scaled if return_logits else F.softmax(scaled, dim=1)

class OsteoarthritisClassificationModel:
    """Production-ready osteoarthritis classification model."""
    def __init__(self, ckpt_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ['Normal', 'Doubtful', 'Mild', 'Moderate', 'Severe']
        self.class_descriptions = {
            'Normal': 'No signs of osteoarthritis',
            'Doubtful': 'Possible early osteoarthritis changes',
            'Mild': 'Mild osteoarthritis with minor joint changes',
            'Moderate': 'Moderate osteoarthritis with clear joint degeneration',
            'Severe': 'Severe osteoarthritis with significant joint damage',
        }

        if ckpt_path and Path(ckpt_path).exists():
            self.model = self._load_calibrated_ensemble(ckpt_path)
        else:
            st.warning("No trained ensemble found. Using demo ResNet-50 model.")
            self.model = self._create_demo_model(num_classes=len(self.class_names))

        self.model.eval()
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel RGB
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _create_demo_model(self, num_classes: int):
        m = tv_models.resnet50(weights="IMAGENET1K_V2")
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        self._using_demo = True
        return m.to(self.device)

    def _load_calibrated_ensemble(self, ckpt_path: str) -> nn.Module:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model_cfgs = ckpt["model_configs"]
        base_models = []
        for cfg in model_cfgs:
            name = cfg["model_name"]
            builder = MODEL_BUILDERS.get(name.lower())
            if builder is None:
                raise ValueError(f"Unknown model_name '{name}' in checkpoint.")
            net = builder(cfg)
            sd = ckpt["individual_models"][len(base_models)]
            net.load_state_dict(sd)
            base_models.append(net)

        weights = np.array(ckpt["ensemble_weights"])
        temperature = float(ckpt["temperature"])
        ensemble = EnsembleModel(base_models, weights, device=self.device)
        calibrated = TemperatureScaling(ensemble, temperature, device=self.device)
        return calibrated

    def predict(self, image: Image.Image) -> Dict:
        """Single-image prediction."""
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x, return_logits=True)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs))

        pred_name = self.class_names[idx]
        return {
            "predicted_class": pred_name,
            "predicted_index": idx,
            "confidence": float(probs[idx]),
            "all_probabilities": {cls: float(p) for cls, p in zip(self.class_names, probs)},
            "description": self.class_descriptions[pred_name],
        }

    def predict_batch(self, images: List[Image.Image]) -> List[Dict]:
        return [self.predict(img) for img in images]

# === Treatment Planning System ===

class ClinicalTreatmentPlanner:
    """AI-powered treatment planning system for osteoarthritis."""
    def __init__(self, api_key: Optional[str] = None):
        self.llm_available = OPENAI_AVAILABLE and api_key is not None
        if self.llm_available:
            openai.api_key = api_key

    def generate_treatment_plan(self, classification_result: Dict, patient_data: Dict) -> Dict:
        if self.llm_available:
            try:
                return self._generate_llm_treatment_plan(classification_result, patient_data)
            except Exception as e:
                st.error(f"LLM Error: {e}")
        return self._generate_rule_based_treatment_plan(classification_result, patient_data)

    def _generate_llm_treatment_plan(self, classification_result: Dict, patient_data: Dict) -> Dict:
        prompt = self._construct_clinical_prompt(classification_result, patient_data)
        text = chat_completion(
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1500,
            temperature=0.3,
            model="gpt-4o-mini",
        )
        return self._parse_treatment_response(text)

    def _generate_rule_based_treatment_plan(self, classification_result: Dict, patient_data: Dict) -> Dict:
        severity = classification_result["predicted_index"]
        base_plan = self._severity_templates().get(severity, self._severity_templates()[2])
        personalized = self._personalize_treatment_plan(base_plan, patient_data)
        return {
            "severity_level": classification_result["predicted_class"],
            "confidence": classification_result["confidence"],
            "primary_approach": personalized["primary"],
            "medications": personalized["medications"],
            "non_pharmacological": personalized["non_pharmacological"],
            "lifestyle": personalized["lifestyle"],
            "surgical_options": personalized.get("surgical_options", []),
            "follow_up": self._get_follow_up_recommendations(severity),
            "red_flags": self._get_red_flags(),
            "references": self._get_clinical_references(),
        }

    def _severity_templates(self) -> Dict[int, Dict]:
        return {
            0: {"primary": "Preventive care", "medications": ["No medications needed"], 
                "non_pharmacological": ["Regular exercise", "Weight management"], 
                "lifestyle": ["Healthy weight", "Regular activity"]},
            1: {"primary": "Early intervention", "medications": ["Topical NSAIDs if symptomatic"],
                "non_pharmacological": ["Physical therapy", "Low-impact exercise"],
                "lifestyle": ["Weight loss if BMI >25", "Anti-inflammatory diet"]},
            2: {"primary": "Conservative management", "medications": ["Topical NSAIDs", "Oral NSAIDs"],
                "non_pharmacological": ["Structured exercise", "Physical therapy"],
                "lifestyle": ["Mediterranean diet", "Weight management"]},
            3: {"primary": "Multimodal pain management", "medications": ["NSAIDs", "Corticosteroid injections"],
                "non_pharmacological": ["Comprehensive physiotherapy", "CBT for pain"],
                "lifestyle": ["Structured weight loss", "Aquatic therapy"]},
            4: {"primary": "Advanced pain management", "medications": ["Combination analgesics", "Intra-articular treatments"],
                "non_pharmacological": ["Multidisciplinary pain management", "Pre-surgical physiotherapy"],
                "surgical_options": ["Total knee replacement", "Partial replacement"],
                "lifestyle": ["Joint protection strategies", "Smoking cessation"]}
        }

    def _personalize_treatment_plan(self, base_plan: Dict, patient_data: Dict) -> Dict:
        plan = {k: v[:] if isinstance(v, list) else v for k, v in base_plan.items()}
        age = patient_data.get("age", 50)
        gender = patient_data.get("gender", "Other")
        comorbidities = [c.lower() for c in patient_data.get("comorbidities", [])]

        if age > 65:
            plan["medications"] = [m for m in plan["medications"] if "NSAIDs" not in m] + [
                "Avoid NSAIDs (↑ GI/CV risk)", "Acetaminophen preferred"
            ]
        if gender.lower() == "female" and age > 50:
            plan["lifestyle"].append("Bone health evaluation (post-menopausal)")
        if "cardiovascular" in comorbidities:
            plan["medications"] = [m for m in plan["medications"] if "NSAIDs" not in m]
            plan["medications"].append("Avoid NSAIDs due to CV risk")

        return plan

    def _get_follow_up_recommendations(self, severity: int) -> List[str]:
        follow_up = {
            0: ["Annual screening"], 1: ["6-month follow-up"], 2: ["3-month follow-up"],
            3: ["6-week follow-up"], 4: ["2-week follow-up", "Surgical consult"]
        }
        return follow_up.get(severity, follow_up[2])

    def _get_red_flags(self) -> List[str]:
        return ["Severe, uncontrolled pain", "Signs of infection", "Significant functional decline"]

    def _get_clinical_references(self) -> List[str]:
        return ["ACR/AF 2019 Guideline", "OARSI Guidelines", "NICE Guideline: Osteoarthritis"]

    def _construct_clinical_prompt(self, classification_result: Dict, patient_data: Dict) -> str:
        return f"""
Clinical Case Assessment:
- X-ray Classification: {classification_result['predicted_class']} ({classification_result['description']})
- AI Confidence: {classification_result['confidence']:.2%}

Patient Demographics:
- Age: {patient_data.get('age', 'Unknown')}
- Gender: {patient_data.get('gender', 'Unknown')}
- Occupation: {patient_data.get('occupation', 'Unknown')}
- Activity Level: {patient_data.get('activity_level', 'Unknown')}
- BMI: {patient_data.get('bmi', 'Unknown')}

Clinical Presentation:
- Current Symptoms: {', '.join(patient_data.get('symptoms', ['None reported']))}
- Treatment Expectations: {patient_data.get('expectations', 'Unknown')}
- Comorbidities: {', '.join(patient_data.get('comorbidities', ['None reported']))}

Please provide a comprehensive, evidence-based treatment plan following current ACR, OARSI, and NICE guidelines. 
Consider the patient's occupation, activity level, and specific symptoms when making recommendations.
Include both pharmacological and non-pharmacological interventions, lifestyle modifications, and follow-up care.
"""

    def _get_system_prompt(self) -> str:
        return (
            "You are an expert rheumatologist and orthopedic specialist. "
            "Provide evidence-based treatment recommendations for osteoarthritis "
            "following current ACR, OARSI, and NICE guidelines. Focus on holistic care."
        )

    def _parse_treatment_response(self, response: str) -> Dict:
        return {
            "llm_response": response,
            "generated_by": "LLM",
            "timestamp": datetime.datetime.now().isoformat(),
        }

# === PDF Generation Functions ===

def generate_patient_letter(patient_info: Dict, analysis_result: Dict, treatment_plan: Dict) -> str:
    """Generate a personalized patient explanation letter using GPT-4."""
    api_key = get_openai_key()
    if not api_key or not OPENAI_AVAILABLE:
        return "Warning: OpenAI API key not configured. Please set OPENAI_API_KEY for AI-generated patient letters."
    
    openai.api_key = api_key
    
    # Build comprehensive patient context from metadata
    patient_context = f"""
Patient Information:
- Name: {patient_info.get('name', 'Patient')}
- Age: {patient_info.get('age', 50)} years old
- Gender: {patient_info.get('gender', 'Unknown')}
- Occupation: {patient_info.get('occupation', 'Not specified')}
- BMI: {patient_info.get('bmi', 'Not specified')}
- Activity Level: {patient_info.get('activity_level', 'Not specified')}
- Current Symptoms: {', '.join(patient_info.get('symptoms', ['Not specified']))}
- Treatment Expectations: {patient_info.get('expectations', 'Not specified')}
- Comorbidities: {', '.join(patient_info.get('comorbidities', ['None reported']))}
- Medical History: {', '.join(patient_info.get('medical_history', ['None reported']))}
- Current Medications: {', '.join(patient_info.get('medications', ['None reported']))}

Analysis Results:
- AI Prediction: {analysis_result['predicted_class']} osteoarthritis
- Confidence Level: {analysis_result['confidence']:.1%}
- Description: {analysis_result['description']}

Treatment Plan Summary:
- Primary Approach: {treatment_plan.get('primary_approach', 'Conservative management')}
- Key Medications: {', '.join(treatment_plan.get('medications', ['None specified'])[:3])}
- Main Lifestyle Changes: {', '.join(treatment_plan.get('lifestyle', ['None specified'])[:3])}
"""

    prompt = f"""
You are a compassionate medical AI assistant helping to explain osteoarthritis analysis results to patients in clear, empathetic language.

{patient_context}

Please write a warm, personalized letter to this patient that:
1. Addresses them by name and acknowledges their specific occupation and lifestyle
2. Explains their X-ray results in simple, non-medical language
3. Validates their experience and symptoms compassionately  
4. Addresses their treatment expectations realistically and encouragingly
5. Explains what their specific grade means for their daily activities and work
6. Considers their age, gender, and activity level in recommendations
7. Provides encouragement and hope while managing expectations appropriately
8. Summarizes key treatment recommendations in patient-friendly language
9. Emphasizes available support and next steps

IMPORTANT: 
- Address the patient directly by name
- Make it personal by referencing their occupation and specific context
- Use warm, professional tone without medical jargon
- Keep it concise but thorough (300-400 words)
- Focus on empowerment and partnership in care
"""

    try:
        return chat_completion(
            messages=[
                {"role": "system", "content": "You are a compassionate medical communication specialist who writes personalized patient letters."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7,
            model="gpt-4o-mini",
        ).strip()
    except Exception as e:
        return f"Error generating patient letter: {str(e)}. Using rule-based template instead."

def generate_clinical_analysis(analysis_result: Dict, patient_info: Dict) -> str:
    """Generate detailed clinical analysis for healthcare providers."""
    api_key = get_openai_key()
    if not api_key or not OPENAI_AVAILABLE:
        return "Warning: OpenAI API key not configured. Please set OPENAI_API_KEY for AI-generated clinical analysis."
    
    openai.api_key = api_key
    
    prompt = f"""
Generate a comprehensive clinical analysis for healthcare providers reviewing an AI-assisted osteoarthritis assessment.

Patient Profile:
- Age: {patient_info.get('age', 50)} years, {patient_info.get('gender', 'Unknown')} 
- Occupation: {patient_info.get('occupation', 'Not specified')}
- BMI: {patient_info.get('bmi', 'Not specified')}
- Activity Level: {patient_info.get('activity_level', 'Not specified')}
- Symptoms: {', '.join(patient_info.get('symptoms', ['Not documented']))}
- Comorbidities: {', '.join(patient_info.get('comorbidities', ['None reported']))}

AI Analysis Results:
- Classification: {analysis_result['predicted_class']} osteoarthritis
- Confidence: {analysis_result['confidence']:.1%}
- Model Description: {analysis_result['description']}

Please provide a clinical analysis that includes:
1. Assessment of AI prediction reliability based on confidence and patient profile
2. Clinical correlation between imaging findings and patient symptoms
3. Risk factors present in this patient
4. Differential diagnosis considerations
5. Recommendations for clinical validation or additional imaging
6. Treatment pathway appropriateness assessment
7. Follow-up and monitoring recommendations
8. Quality assurance notes for the AI prediction

Format as a structured clinical note appropriate for physician review.
Use professional medical terminology and evidence-based recommendations.
"""

    try:
        return chat_completion(
            messages=[
                {"role": "system", "content": "You are an expert rheumatologist providing clinical analysis of AI-assisted osteoarthritis assessments."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3,
            model="gpt-4o-mini",
        ).strip()
    except Exception as e:
        return f"Error generating clinical analysis: {str(e)}. Manual review required."

def generate_exercise_plan(patient_info: Dict, severity: str) -> str:
    """Generate personalized exercise recommendations using GPT-4."""
    api_key = get_openai_key()
    if not api_key or not OPENAI_AVAILABLE:
        return "Warning: OpenAI API key not configured for exercise plan generation."
    
    openai.api_key = api_key
    
    prompt = f"""
Generate a personalized exercise plan for a patient with {severity} osteoarthritis.

Patient Profile:
- Age: {patient_info.get('age', 50)} years
- Gender: {patient_info.get('gender', 'Unknown')}
- Occupation: {patient_info.get('occupation', 'Not specified')}
- BMI: {patient_info.get('bmi', 'Not specified')}
- Activity Level: {patient_info.get('activity_level', 'Not specified')}
- Current Symptoms: {', '.join(patient_info.get('symptoms', ['Not specified']))}
- Treatment Expectations: {patient_info.get('expectations', 'Not specified')}
- Comorbidities: {', '.join(patient_info.get('comorbidities', ['None reported']))}
- Medical History: {', '.join(patient_info.get('medical_history', ['None reported']))}
- Current Medications: {', '.join(patient_info.get('medications', ['None reported']))}

Please provide:
1. 3-4 specific exercises suitable for their condition and symptoms
2. Frequency and duration for each exercise, considering their occupation and lifestyle
3. Important safety precautions specific to their symptoms and comorbidities
4. Progression guidelines aligned with their expectations
5. When to stop and consult their doctor
6. Modifications based on their current symptoms and activity level
7. How exercises relate to their specific goals and occupation

Make it practical and achievable. Consider their age, BMI, occupation, current activity level, and symptoms.
Address any unrealistic expectations compassionately while providing hope.
Format with clear headings and bullet points.
"""

    try:
        return chat_completion(
            messages=[
                {"role": "system", "content": "You are a physical therapy specialist creating safe, evidence-based exercise plans."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.6,
            model="gpt-4o-mini",
        ).strip()
    except Exception as e:
        return f"Error generating exercise plan: {str(e)}"

def generate_lifestyle_recommendations(patient_info: Dict, severity: str) -> str:
    """Generate personalized lifestyle recommendations using GPT-4."""
    api_key = get_openai_key()
    if not api_key or not OPENAI_AVAILABLE:
        return "Warning: OpenAI API key not configured for lifestyle recommendations."
    
    openai.api_key = api_key
    
    prompt = f"""
Create personalized lifestyle recommendations for a patient with {severity} osteoarthritis.

Patient Details:
- Age: {patient_info.get('age', 50)} years
- Gender: {patient_info.get('gender', 'Unknown')}
- Occupation: {patient_info.get('occupation', 'Not specified')}
- BMI: {patient_info.get('bmi', 'Not specified')}
- Activity Level: {patient_info.get('activity_level', 'Not specified')}
- Current Symptoms: {', '.join(patient_info.get('symptoms', ['Not specified']))}
- Treatment Expectations: {patient_info.get('expectations', 'Not specified')}
- Comorbidities: {', '.join(patient_info.get('comorbidities', ['None reported']))}
- Medical History: {', '.join(patient_info.get('medical_history', ['None reported']))}
- Current Medications: {', '.join(patient_info.get('medications', ['None reported']))}

Focus on:
1. Diet and nutrition advice (especially for joint health and weight management if needed)
2. Sleep and stress management, considering their current symptoms and occupation
3. Daily activity modifications that align with their expectations and work demands
4. Joint protection strategies specific to their symptoms and occupation
5. Pain management techniques that complement their current medications
6. When to seek additional help
7. Lifestyle modifications that support their specific goals and occupation
8. Consideration of medication interactions or side effects that may affect lifestyle choices

Make recommendations specific to their profile, symptoms, occupation, and goals. Be practical and achievable.
Address any unrealistic expectations compassionately while providing hope.
Use encouraging language and explain the "why" behind each recommendation.
"""

    try:
        return chat_completion(
            messages=[
                {"role": "system", "content": "You are a lifestyle medicine specialist focused on osteoarthritis management."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.6,
            model="gpt-4o-mini",
        ).strip()
    except Exception as e:
        return f"Error generating lifestyle recommendations: {str(e)}"

def process_text_with_bullets(text: str, content: list, normal_style, bullet_style):
    """Process text content with bullet points for PDF generation."""
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            from reportlab.platypus import Spacer
            content.append(Spacer(1, 3))
            continue
            
        if line.startswith(('- ', '* ', '■ ')):
            bullet_text = '- ' + line[2:]
            content.append(Paragraph(bullet_text, bullet_style))
        else:
            # Check if line is a header/subheading
            medical_headers = ['Exercise Plan', 'Lifestyle Recommendations', 'Safety Precautions', 'Progression Guidelines',
                              'Diet and Nutrition', 'Sleep and Stress', 'Joint Protection', 'Pain Management',
                              'When to Seek Help', 'Daily Activities', 'Important Notes', 'Warning Signs']
            
            is_header = any(header in line for header in medical_headers) or line.isupper() or line.endswith(':')
            
            if is_header:
                from reportlab.lib.styles import ParagraphStyle
                from reportlab.lib import colors
                subheading_style = ParagraphStyle(
                    'SubHeading', parent=normal_style, fontSize=11, spaceAfter=6,
                    textColor=colors.darkred, fontName='Helvetica-Bold'
                )
                content.append(Paragraph(line, subheading_style))
            else:
                content.append(Paragraph(line, normal_style))

def convert_markdown_to_reportlab_html(text: str) -> str:
    """Convert a safe subset of Markdown to ReportLab-friendly inline tags.

    Supports bold (**) and italics (*). Escapes HTML, then re-inserts allowed tags.
    Ensures proper closing tags to avoid ReportLab parse errors.
    """
    if not text:
        return ""
    # Escape existing HTML special chars first
    safe = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    # Bold: **text**
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    # Italic: *text*
    safe = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", safe)
    # Line breaks
    safe = safe.replace('\n', '<br/>')
    return safe

def create_patient_pdf(patient_info: Dict, analysis_result: Dict, patient_letter: str, 
                      exercise_plan: str = None, lifestyle_recommendations: str = None) -> bytes:
    """Generate a comprehensive patient letter PDF following professional medical format."""
    if not PDF_AVAILABLE:
        raise ImportError("ReportLab not available. Please install: pip install reportlab")
    
    from reportlab.platypus import KeepTogether, PageBreak
    from reportlab.lib.units import inch
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    # Medical report styles
    styles = getSampleStyleSheet()
    
    # Title style for medical reports
    title_style = ParagraphStyle(
        'MedicalTitle', parent=styles['Heading1'], fontSize=18, spaceAfter=30,
        alignment=TA_CENTER, textColor=COLORS['primary']['medical_blue']
    )
    
    # Header style for medical institution
    header_style = ParagraphStyle(
        'MedicalHeader', parent=styles['Normal'], fontSize=12, spaceAfter=20,
        alignment=TA_CENTER, textColor=COLORS['neutral']['charcoal']
    )
    
    # Section heading style
    section_style = ParagraphStyle(
        'SectionHeading', parent=styles['Heading2'], fontSize=14, spaceAfter=12,
        spaceBefore=20, textColor=COLORS['primary']['medical_blue']
    )
    
    # Professional body text
    body_style = ParagraphStyle(
        'MedicalBody', parent=styles['Normal'], fontSize=11, leading=14,
        spaceAfter=8, fontName='Helvetica', justifyBreaks=1
    )
    
    # Important information style
    important_style = ParagraphStyle(
        'Important', parent=styles['Normal'], fontSize=11, leading=14,
        spaceAfter=8, fontName='Helvetica-Bold', textColor=COLORS['semantic']['error_red']
    )
    
    content = []
    
    # MEDICAL HEADER
    content.append(Paragraph("OSTEOARTHRITIS ASSESSMENT REPORT", title_style))
    content.append(Paragraph("AI-Assisted Clinical Analysis", header_style))
    content.append(Spacer(1, 20))
    
    # PATIENT INFORMATION SECTION
    content.append(Paragraph("PATIENT INFORMATION", section_style))
    
    # Create patient info table
    patient_data = [
        ['Report Date:', datetime.datetime.now().strftime('%B %d, %Y')],
        ['Patient Name:', patient_info.get('name', 'Patient')],
        ['Age:', f"{patient_info.get('age', 'Unknown')} years"],
        ['Gender:', patient_info.get('gender', 'Not specified')],
        ['Occupation:', patient_info.get('occupation', 'Not specified')],
    ]
    
    # Add BMI if available
    if patient_info.get('bmi'):
        patient_data.append(['BMI:', str(patient_info['bmi'])])
    
    # Add activity level if available
    if patient_info.get('activity_level'):
        patient_data.append(['Activity Level:', patient_info['activity_level']])
    
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (1,0), (1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, COLORS['neutral']['light_gray']),
    ]))
    content.append(patient_table)
    content.append(Spacer(1, 20))
    
    # CLINICAL FINDINGS SECTION
    content.append(Paragraph("CLINICAL FINDINGS", section_style))
    
    # AI Analysis Results
    findings_data = [
        ['X-ray Classification:', analysis_result['predicted_class']],
        ['Severity Grade:', f"KL Grade {analysis_result.get('predicted_index', 'Unknown')}"],
        ['AI Confidence:', f"{analysis_result['confidence']:.1%}"],
        ['Clinical Description:', analysis_result['description']],
    ]
    
    findings_table = Table(findings_data, colWidths=[2*inch, 4*inch])
    findings_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (1,0), (1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, COLORS['neutral']['light_gray']),
        ('BACKGROUND', (0,0), (-1,0), COLORS['neutral']['off_white']),
    ]))
    content.append(findings_table)
    content.append(Spacer(1, 20))
    
    # PATIENT LETTER SECTION
    content.append(Paragraph("PERSONALIZED HEALTH INFORMATION", section_style))
    
    # Process patient letter with better formatting
    for paragraph in patient_letter.split('\n\n'):
        if paragraph.strip():
            # Check if this is a heading (starts with ###, ##, or #)
            if paragraph.strip().startswith('###'):
                heading_text = paragraph.strip().replace('###', '').strip()
                sub_heading_style = ParagraphStyle(
                    'SubHeading', parent=styles['Heading3'], fontSize=12, spaceAfter=8,
                    spaceBefore=12, textColor=COLORS['primary']['healthcare_teal']
                )
                content.append(Paragraph(heading_text, sub_heading_style))
            elif paragraph.strip().startswith('##'):
                heading_text = paragraph.strip().replace('##', '').strip()
                content.append(Paragraph(heading_text, section_style))
            elif paragraph.strip().startswith('#'):
                heading_text = paragraph.strip().replace('#', '').strip()
                content.append(Paragraph(heading_text, section_style))
            else:
                formatted_paragraph = convert_markdown_to_reportlab_html(paragraph.strip())
                content.append(Paragraph(formatted_paragraph, body_style))
    
    content.append(Spacer(1, 20))
    
    # PAGE BREAK FOR EXERCISE PLAN
    if exercise_plan:
        content.append(PageBreak())
        content.append(Paragraph("RECOMMENDED EXERCISE PROGRAM", section_style))
        
        content.append(Paragraph(
            "This exercise program has been specifically designed based on your osteoarthritis assessment. "
            "Please follow the guidelines carefully and consult with your healthcare provider before starting any new exercise routine.",
            body_style
        ))
        content.append(Spacer(1, 12))
        
        process_text_with_bullets(exercise_plan, content, body_style, body_style)
    
    # PAGE BREAK FOR LIFESTYLE RECOMMENDATIONS
    if lifestyle_recommendations:
        content.append(PageBreak())
        content.append(Paragraph("LIFESTYLE RECOMMENDATIONS", section_style))
        
        content.append(Paragraph(
            "These lifestyle modifications can help manage your osteoarthritis symptoms and improve your overall quality of life:",
            body_style
        ))
        content.append(Spacer(1, 12))
        
        process_text_with_bullets(lifestyle_recommendations, content, body_style, body_style)
    
    # IMPORTANT DISCLAIMERS
    content.append(Spacer(1, 30))
    content.append(Paragraph("IMPORTANT MEDICAL DISCLAIMER", section_style))
    
    disclaimer_text = """
    This AI-assisted analysis is provided for informational purposes and to support clinical decision-making. 
    It is not intended to replace professional medical judgment, diagnosis, or treatment. Always consult with 
    your qualified healthcare provider for proper medical advice, diagnosis, and treatment options specific to your condition.
    
    The AI analysis should be interpreted in conjunction with your complete medical history, physical examination, 
    and other relevant clinical factors by a qualified healthcare professional.
    """
    
    content.append(Paragraph(disclaimer_text, important_style))
    
    # FOOTER
    content.append(Spacer(1, 20))
    footer_style = ParagraphStyle(
        'Footer', parent=styles['Normal'], fontSize=9, leading=11,
        alignment=TA_CENTER, textColor=COLORS['neutral']['slate_gray']
    )
    
    content.append(Paragraph(
        f"Generated on {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')} | "
        "Osteoarthritis Clinical Decision Support System",
        footer_style
    ))
    
    # Build the PDF
    doc.build(content)
    buffer.seek(0)
    return buffer.getvalue()

# === Model Loading and Prediction Functions ===

def load_model(model_path: Path = None, use_ensemble: bool = True) -> OsteoarthritisClassificationModel:
    """Load the trained model for inference."""
    if model_path and model_path.exists():
        try:
            classifier = OsteoarthritisClassificationModel(model_path=str(model_path), use_ensemble=use_ensemble)
            st.success(f"Model loaded: {model_path.name}")
            return classifier
        except Exception as e:
            st.warning(f"Could not load model from {model_path}: {str(e)}")
    
    # Load demo patients for validation
    try:
        classifier = OsteoarthritisClassificationModel(use_ensemble=use_ensemble)
        st.info("Using demo classification model")
        return classifier
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        st.stop()

@st.cache_resource
def get_model():
    """Get cached model instance."""
    models_dir = REPO_ROOT / "models" / "deployment"
    model_files = list(models_dir.glob("*.pth")) if models_dir.exists() else []
    
    if model_files:
        # Use the most recent model file
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        return load_model(latest_model, use_ensemble=True)
    else:
        return load_model(use_ensemble=True)

def create_clinical_pdf(patient_info: Dict, analysis_result: Dict, clinical_analysis: str, treatment_plan: Dict, advanced_analysis: Dict = None) -> bytes:
    """Generate a professional clinical review PDF for healthcare providers."""
    if not PDF_AVAILABLE:
        raise ImportError("ReportLab not available. Please install: pip install reportlab")
    
    from reportlab.platypus import KeepTogether, PageBreak
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    
    # Professional clinical styles
    styles = getSampleStyleSheet()
    
    # Title style for clinical reports
    title_style = ParagraphStyle(
        'ClinicalTitle', parent=styles['Heading1'], fontSize=18, spaceAfter=20,
        alignment=TA_CENTER, textColor=COLORS['primary']['medical_blue'], fontName='Helvetica-Bold'
    )
    
    # Header for institution/clinic
    institution_style = ParagraphStyle(
        'Institution', parent=styles['Normal'], fontSize=11, spaceAfter=15,
        alignment=TA_CENTER, textColor=COLORS['neutral']['charcoal']
    )
    
    # Section headings
    section_style = ParagraphStyle(
        'ClinicalSection', parent=styles['Heading2'], fontSize=14, spaceAfter=10,
        spaceBefore=15, textColor=COLORS['primary']['medical_blue'], fontName='Helvetica-Bold'
    )
    
    # Subsection headings
    subsection_style = ParagraphStyle(
        'ClinicalSubsection', parent=styles['Heading3'], fontSize=12, spaceAfter=8,
        spaceBefore=10, textColor=COLORS['primary']['healthcare_teal'], fontName='Helvetica-Bold'
    )
    
    # Professional body text
    body_style = ParagraphStyle(
        'ClinicalBody', parent=styles['Normal'], fontSize=10, leading=13,
        spaceAfter=6, fontName='Helvetica'
    )
    
    # Important clinical notes
    clinical_note_style = ParagraphStyle(
        'ClinicalNote', parent=styles['Normal'], fontSize=10, leading=13,
        spaceAfter=6, fontName='Helvetica-Bold', textColor=COLORS['semantic']['error_red']
    )
    
    # Review status style
    review_style = ParagraphStyle(
        'ReviewStatus', parent=styles['Normal'], fontSize=12, leading=15,
        spaceAfter=8, fontName='Helvetica-Bold', textColor=COLORS['semantic']['warning_orange']
    )
    
    content = []
    
    # CLINICAL HEADER
    content.append(Paragraph("CLINICAL REVIEW REPORT", title_style))
    content.append(Paragraph("AI-Assisted Osteoarthritis Assessment", institution_style))
    content.append(Paragraph("FOR HEALTHCARE PROVIDER REVIEW", institution_style))
    content.append(Spacer(1, 20))
    
    # REVIEW STATUS BOX
    content.append(Paragraph("REVIEW STATUS", section_style))
    review_data = [
        ['Report Generated:', datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')],
        ['Reviewing Physician:', '________________________________'],
        ['Clinical Approval:', '☐ APPROVED    ☐ NEEDS REVISION    ☐ REJECTED'],
        ['Date Reviewed:', '________________'],
        ['Signature:', '________________________________'],
    ]
    
    review_table = Table(review_data, colWidths=[2*inch, 4*inch])
    review_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (1,0), (1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 1, COLORS['neutral']['charcoal']),
        ('BACKGROUND', (0,0), (-1,0), COLORS['neutral']['light_gray']),
    ]))
    content.append(review_table)
    content.append(Spacer(1, 20))
    
    # PATIENT SUMMARY SECTION
    content.append(Paragraph("PATIENT SUMMARY", section_style))
    
    # Comprehensive patient information
    patient_summary_data = [
        ['Patient Name:', patient_info.get('name', 'Patient')],
        ['Age:', f"{patient_info.get('age', 'Unknown')} years"],
        ['Gender:', patient_info.get('gender', 'Not specified')],
        ['Date of Birth:', patient_info.get('dob', 'Not specified')],
        ['Occupation:', patient_info.get('occupation', 'Not specified')],
        ['BMI:', str(patient_info.get('bmi', 'Not specified'))],
        ['Activity Level:', patient_info.get('activity_level', 'Not specified')],
    ]
    
    patient_summary_table = Table(patient_summary_data, colWidths=[2*inch, 4*inch])
    patient_summary_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (1,0), (1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, COLORS['neutral']['light_gray']),
    ]))
    content.append(patient_summary_table)
    content.append(Spacer(1, 15))
    
    # CLINICAL HISTORY
    content.append(Paragraph("CLINICAL HISTORY", subsection_style))
    
    symptoms_str = ', '.join(patient_info.get('symptoms', ['Not documented']))
    comorbidities_str = ', '.join(patient_info.get('comorbidities', ['None reported']))
    medical_history_str = ', '.join(patient_info.get('medical_history', ['None documented']))
    medications_str = ', '.join(patient_info.get('medications', ['None reported']))
    
    clinical_history_data = [
        ['Current Symptoms:', symptoms_str],
        ['Comorbidities:', comorbidities_str],
        ['Medical History:', medical_history_str],
        ['Current Medications:', medications_str],
        ['Treatment Expectations:', patient_info.get('expectations', 'Not specified')],
    ]
    
    clinical_history_table = Table(clinical_history_data, colWidths=[2*inch, 4*inch])
    clinical_history_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (1,0), (1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, COLORS['neutral']['light_gray']),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    content.append(clinical_history_table)
    content.append(Spacer(1, 20))
    
    # AI ANALYSIS RESULTS
    content.append(Paragraph("AI ANALYSIS RESULTS", section_style))
    
    # Map predicted_class to KL Grade if it contains numbers
    predicted_class = analysis_result['predicted_class']
    kl_grade = "Unknown"
    if any(char.isdigit() for char in predicted_class):
        for i, char in enumerate(predicted_class):
            if char.isdigit():
                kl_grade = f"KL Grade {char}"
                break
    
    ai_results_data = [
        ['X-ray Classification:', predicted_class],
        ['Kellgren-Lawrence Grade:', kl_grade],
        ['AI Model Confidence:', f"{analysis_result['confidence']:.1%}"],
        ['Clinical Description:', analysis_result['description']],
        ['Model Architecture:', 'Deep Learning Ensemble'],
        ['Analysis Timestamp:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Quality Assurance:', '☐ Image quality adequate    ☐ Positioning acceptable'],
        ['Clinical Correlation:', '☐ Consistent with symptoms    ☐ Inconsistent - review needed'],
    ]
    
    ai_results_table = Table(ai_results_data, colWidths=[2*inch, 4*inch])
    ai_results_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (1,0), (1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 1, COLORS['neutral']['charcoal']),
        ('BACKGROUND', (0,0), (-1,0), COLORS['neutral']['off_white']),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    content.append(ai_results_table)
    content.append(Spacer(1, 20))
    
    # CLINICAL ASSESSMENT SECTION (NEW PAGE)
    content.append(PageBreak())
    content.append(Paragraph("CLINICAL ASSESSMENT", section_style))
    
    # Process clinical analysis
    content.append(Paragraph("AI-Generated Clinical Analysis:", subsection_style))
    for paragraph in clinical_analysis.split('\n\n'):
        if paragraph.strip():
            formatted_paragraph = convert_markdown_to_reportlab_html(paragraph.strip())
            content.append(Paragraph(formatted_paragraph, body_style))
    
    content.append(Spacer(1, 15))
    
    # TREATMENT PLAN REVIEW
    content.append(Paragraph("TREATMENT PLAN ASSESSMENT", subsection_style))
    
    if treatment_plan:
        # Extract key treatment components
        primary_approach = treatment_plan.get('primary_approach', 'Not specified')
        medications = treatment_plan.get('medications', [])
        non_pharm = treatment_plan.get('non_pharmacological', [])
        
        content.append(Paragraph(f"<b>Primary Treatment Approach:</b> {primary_approach}", body_style))
        
        if medications:
            content.append(Paragraph("<b>Pharmacological Interventions:</b>", body_style))
            for med in medications[:3]:  # Limit to first 3 for space
                content.append(Paragraph(f"- {med}", body_style))
        
        if non_pharm:
            content.append(Paragraph("<b>Non-Pharmacological Interventions:</b>", body_style))
            for intervention in non_pharm[:3]:  # Limit to first 3 for space
                content.append(Paragraph(f"- {intervention}", body_style))
    
    content.append(Spacer(1, 15))
    
    # CLINICAL DECISION SUPPORT
    content.append(Paragraph("CLINICAL DECISION SUPPORT", subsection_style))
    
    clinical_decision_data = [
        ['Confidence Assessment:', '☐ High (>90%)    ☐ Moderate (70-90%)    ☐ Low (<70%)'],
        ['Requires Additional Imaging:', '☐ Yes    ☐ No'],
        ['Specialist Referral Needed:', '☐ Rheumatology    ☐ Orthopedics    ☐ Pain Management    ☐ None'],
        ['Treatment Plan Approval:', '☐ Approve as suggested    ☐ Modify    ☐ Create new plan'],
        ['Follow-up Interval:', '☐ 2 weeks    ☐ 4 weeks    ☐ 3 months    ☐ 6 months'],
        ['Patient Education Provided:', '☐ Yes    ☐ No    ☐ Scheduled'],
    ]
    
    clinical_decision_table = Table(clinical_decision_data, colWidths=[2.5*inch, 3.5*inch])
    clinical_decision_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (1,0), (1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 1, COLORS['neutral']['charcoal']),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    content.append(clinical_decision_table)
    content.append(Spacer(1, 20))
    
    # CLINICAL NOTES SECTION
    content.append(Paragraph("CLINICAL NOTES", subsection_style))
    content.append(Paragraph("Additional clinical observations and modifications:", body_style))
    
    # Create lined space for handwritten notes
    for i in range(5):
        content.append(Spacer(1, 15))
        # Create a line for writing
        line_data = [['_' * 80]]
        line_table = Table(line_data, colWidths=[6*inch])
        line_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ]))
        content.append(line_table)
    
    content.append(Spacer(1, 20))
    
    # RED FLAGS AND WARNINGS
    content.append(Paragraph("CLINICAL RED FLAGS", subsection_style))
    content.append(Paragraph(
        "Monitor for: Severe uncontrolled pain, signs of infection, significant functional decline, "
        "neurological symptoms, inability to bear weight, suspected fracture",
        clinical_note_style
    ))
    
    content.append(Spacer(1, 15))
    
    # FOOTER AND SIGNATURES
    content.append(Paragraph("CLINICAL APPROVAL", subsection_style))
    
    approval_data = [
        ['Physician Name:', '_' * 30],
        ['Medical License #:', '_' * 20],
        ['Signature:', '_' * 30],
        ['Date:', '_' * 15],
        ['Next Review Date:', '_' * 15],
    ]
    
    approval_table = Table(approval_data, colWidths=[2*inch, 4*inch])
    approval_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (1,0), (1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 1, COLORS['neutral']['charcoal']),
    ]))
    content.append(approval_table)
    
    # FINAL DISCLAIMER
    content.append(Spacer(1, 20))
    disclaimer_style = ParagraphStyle(
        'Disclaimer', parent=styles['Normal'], fontSize=8, leading=10,
        alignment=TA_CENTER, textColor=COLORS['neutral']['slate_gray']
    )
    
    content.append(Paragraph(
        "This AI-assisted analysis is intended to support clinical decision-making and must be reviewed by a qualified healthcare provider. "
        "The final diagnosis and treatment decisions remain the responsibility of the attending physician.",
        disclaimer_style
    ))
    
    content.append(Paragraph(
        f"Generated by Osteoarthritis Clinical Decision Support System | "
        f"Report ID: {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        disclaimer_style
    ))
    
    # Build the PDF
    doc.build(content)
    buffer.seek(0)
    return buffer.getvalue()

# === Streamlit Application ===

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Osteoarthritis Clinical Decision Support",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # CSS styling
    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;
        }
        .prediction-box {
            border: 2px solid #2a5298; border-radius: 10px; padding: 1rem; margin: 1rem 0; background-color: #f8f9fa;
        }
        .treatment-section {
            border-left: 4px solid #28a745; padding-left: 1rem; margin: 1rem 0;
        }
        .warning-box {
            background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 1rem; margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

    # Header
    st.markdown("""
        <div class="main-header">
            <h1>Osteoarthritis Clinical Decision Support System</h1>
            <p>AI-Powered Diagnosis and Evidence-Based Treatment Planning</p>
        </div>
        """, unsafe_allow_html=True)

    # Initialize session state
    if "model" not in st.session_state:
        with st.spinner("Loading AI model..."):
            default_ckpt = REPO_ROOT / "models" / "deployment" / "best_model_for_deployment.pth"
            try:
                st.session_state.model = OsteoarthritisClassificationModel(str(default_ckpt))
            except Exception as e:
                st.error(f"Failed to load model from {default_ckpt}: {e}")
                st.warning("Using default demo model instead.")
                st.session_state.model = OsteoarthritisClassificationModel()

    if "treatment_planner" not in st.session_state:
        api_key = get_openai_key()
        st.session_state.treatment_planner = ClinicalTreatmentPlanner(api_key)
        if api_key:
            masked = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
            st.success(f"AI-powered treatment recommendations enabled (key: {masked})", icon="✅")
        else:
            st.warning("OpenAI key not detected. Configure OPENAI_API_KEY in .streamlit/secrets.toml or environment.")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose Function", 
        ["Single Patient Analysis", "Batch Processing", "Patient Dashboard", "Analytics Dashboard", "Settings"])

    if page == "Single Patient Analysis":
        single_patient_interface()
    elif page == "Batch Processing":
        batch_processing_interface()
    elif page == "Patient Dashboard":
        patient_dashboard_interface()
    elif page == "Analytics Dashboard":
        analytics_dashboard()
    elif page == "Settings":
        settings_interface()

def single_patient_interface():
    st.header("Single Patient Analysis")
    
    # Demo Patients Quick Load Section
    st.subheader("Demo Patients")
    demo_patients_dir = REPO_ROOT / "data" / "consensus" / "demo_patients"
    
    if demo_patients_dir.exists():
        # Load demo patient metadata
        metadata_file = demo_patients_dir / "demo_patients_metadata.json"
        demo_metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    demo_metadata = json.load(f)
            except:
                pass
        
        # Create demo patient selection
        demo_options = ["Select a demo patient..."]
        demo_files = {}
        
        for severity_dir in sorted(demo_patients_dir.iterdir()):
            if severity_dir.is_dir():
                for img_file in severity_dir.glob("*.png"):
                    if img_file.name in demo_metadata:
                        patient_info = demo_metadata[img_file.name]
                        # Don't reveal the actual severity class in the selection
                        display_name = f"{patient_info['name']} ({patient_info['age']}{patient_info['gender'][0]}) - {patient_info['occupation']}"
                        demo_options.append(display_name)
                        # Store actual severity from folder name for comparison
                        actual_severity = severity_dir.name[1:]  # Remove the leading number
                        demo_files[display_name] = (img_file, patient_info, actual_severity)
        
        selected_demo = st.selectbox("Quick Load Demo Patient", demo_options)
        
        if selected_demo != "Select a demo patient..." and selected_demo in demo_files:
            demo_file, patient_info, actual_severity = demo_files[selected_demo]
            
            # Load and display demo patient (without revealing actual classification)
            st.info(f"Loaded: **{patient_info['name']}** - {patient_info['occupation']}")
            
            # Auto-populate patient information from metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                patient_age = st.number_input("Age", min_value=18, max_value=120, value=patient_info['age'])
                patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"], 
                                             index=["Male", "Female", "Other"].index(patient_info['gender']))
            with col2:
                symptom_options = [
                    "Joint pain", "Stiffness", "Swelling", "Reduced mobility",
                    "Grinding sensation", "Joint instability", "Morning stiffness"
                ]
                # Smart mapping from patient metadata to dropdown options
                patient_symptoms = patient_info.get('symptoms', [])
                symptom_default = []
                for symptom in patient_symptoms:
                    symptom_lower = symptom.lower()
                    if any(word in symptom_lower for word in ['pain', 'ache', 'hurt']):
                        if "Joint pain" not in symptom_default:
                            symptom_default.append("Joint pain")
                    if any(word in symptom_lower for word in ['stiff', 'morning stiff']):
                        if "stiffness" in symptom_lower and "morning" in symptom_lower:
                            if "Morning stiffness" not in symptom_default:
                                symptom_default.append("Morning stiffness")
                        elif "Stiffness" not in symptom_default:
                            symptom_default.append("Stiffness")
                    if any(word in symptom_lower for word in ['swell', 'inflam']):
                        if "Swelling" not in symptom_default:
                            symptom_default.append("Swelling")
                    if any(word in symptom_lower for word in ['mobility', 'movement', 'difficulty', 'limitation']):
                        if "Reduced mobility" not in symptom_default:
                            symptom_default.append("Reduced mobility")
                
                symptoms = st.multiselect("Current Symptoms", symptom_options, default=symptom_default)
            with col3:
                expectation_options = [
                    "Pain relief", "Improved mobility", "Prevent progression", "Return to activities", "Surgery avoidance"
                ]
                # Smart mapping for treatment expectations
                patient_expectation = patient_info.get('treatment_expectations', '').lower()
                expectation_index = 0  # Default to "Pain relief"
                
                if any(word in patient_expectation for word in ['prevent', 'progression', 'early']):
                    expectation_index = 2  # "Prevent progression"
                elif any(word in patient_expectation for word in ['mobility', 'active', 'movement', 'function']):
                    expectation_index = 1  # "Improved mobility"  
                elif any(word in patient_expectation for word in ['pain', 'relief', 'manage']):
                    expectation_index = 0  # "Pain relief"
                elif any(word in patient_expectation for word in ['work', 'activity', 'return', 'continue']):
                    expectation_index = 3  # "Return to activities"
                elif any(word in patient_expectation for word in ['avoid', 'surgery', 'non-surgical']):
                    expectation_index = 4  # "Surgery avoidance"
                
                expectations = st.selectbox("Treatment Expectations", expectation_options, index=expectation_index)
                comorbidity_options = [
                    "Diabetes", "Cardiovascular disease", "Hypertension", "Kidney disease"
                ]
                # Smart mapping for comorbidities
                patient_comorbidities = patient_info.get('comorbidities', [])
                comorbidity_default = []
                for condition in patient_comorbidities:
                    condition_lower = condition.lower()
                    if any(word in condition_lower for word in ['diabetes', 'diabetic']):
                        if "Diabetes" not in comorbidity_default:
                            comorbidity_default.append("Diabetes")
                    if any(word in condition_lower for word in ['hypertension', 'high blood pressure']):
                        if "Hypertension" not in comorbidity_default:
                            comorbidity_default.append("Hypertension")
                    if any(word in condition_lower for word in ['cardiovascular', 'heart', 'cardiac']):
                        if "Cardiovascular disease" not in comorbidity_default:
                            comorbidity_default.append("Cardiovascular disease")
                    if any(word in condition_lower for word in ['kidney', 'renal']):
                        if "Kidney disease" not in comorbidity_default:
                            comorbidity_default.append("Kidney disease")
                
                comorbidities = st.multiselect("Comorbidities", comorbidity_options, default=comorbidity_default)
            
            # Auto-load the demo image
            demo_image = Image.open(demo_file)
            
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(demo_image, caption=f"Demo Patient: {patient_info['name']}", use_container_width=True)
                
                # Display patient context
                st.markdown("**Patient Context**")
                st.write(f"**Occupation:** {patient_info['occupation']}")
                st.write(f"**BMI:** {patient_info.get('bmi', 'N/A')}")
                st.write(f"**Activity Level:** {patient_info.get('activity_level', 'N/A')}")
                if patient_info.get('medical_history'):
                    st.write(f"**Medical History:** {', '.join(patient_info['medical_history'])}")
                if patient_info.get('medications'):
                    st.write(f"**Current Medications:** {', '.join(patient_info['medications'])}")

            with c2:
                with st.spinner("Analyzing demo patient X-ray (blind prediction)..."):
                    # Make blind prediction without knowing actual classification
                    prediction = st.session_state.model.predict(demo_image)

                st.markdown('<div class="prediction-box"><h3>AI Analysis Results</h3></div>', unsafe_allow_html=True)
                
                # Show AI prediction
                ai_severity = prediction["predicted_class"]
                confidence = prediction["confidence"]
                st.metric("AI Predicted Severity", ai_severity, delta=f"Confidence: {confidence:.1%}")
                st.write(f"**AI Assessment:** {prediction['description']}")
                
                # Show comparison with actual classification
                st.markdown("---")
                st.subheader("Prediction Validation")
                
                col_pred, col_actual, col_match = st.columns(3)
                with col_pred:
                    st.metric("AI Prediction", ai_severity)
                with col_actual:
                    st.metric("Actual Classification", actual_severity)
                with col_match:
                    is_correct = ai_severity == actual_severity
                    match_status = "Correct" if is_correct else "Incorrect"
                    match_color = "success" if is_correct else "error"
                    if is_correct:
                        st.success(f"**{match_status}**")
                    else:
                        st.error(f"**{match_status}**")
                
                # Show detailed comparison
                with st.expander("Detailed Classification Analysis"):
                    st.write(f"**Patient:** {patient_info['name']}")
                    st.write(f"**AI Prediction:** {ai_severity} (Confidence: {confidence:.1%})")
                    st.write(f"**Ground Truth:** {actual_severity}")
                    
                    if is_correct:
                        st.success("Correct classification - AI prediction matches expert annotation")
                    else:
                        st.error("Misclassification detected")
                        st.write("**Clinical Implications:**")
                        st.write(f"- AI predicted **{ai_severity}** severity")
                        st.write(f"- Actual severity is **{actual_severity}**")
                        st.write("- Treatment recommendations are based on AI prediction")
                        st.write("- Consider this discrepancy in clinical decision-making")

                st.subheader("Probability Distribution")
                prob_data = prediction["all_probabilities"]
                fig = px.bar(x=list(prob_data.keys()), y=list(prob_data.values()),
                            title="AI Classification Probabilities",
                            labels={"x": "Severity Level", "y": "Probability"})
                
                # Highlight the actual classification in the chart
                colors = ['red' if severity == actual_severity else 'lightblue' for severity in prob_data.keys()]
                fig.update_traces(marker_color=colors)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Add annotation about the highlighted bar
                st.caption(f"🔴 Red bar shows actual classification ({actual_severity})")

            # Treatment plan generated automatically - see streamlined workflow below
            
            # === STREAMLINED WORKFLOW: Complete AI Analysis & Documentation ===
            st.subheader("Complete AI Clinical Documentation")
            st.info("Generate all documentation with a single click: treatment plan, patient letter, and clinical report")
            
            # Generate unique case ID if not exists
            if 'case_id' not in st.session_state:
                st.session_state.case_id = f"CASE_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            st.info(f"**Case ID:** {st.session_state.case_id}")
            
            # Check for OpenAI API key
            api_key_available = get_openai_key()
            
            if not api_key_available:
                st.warning("OpenAI API Key required: set OPENAI_API_KEY environment variable to enable AI-generated documentation.")
                with st.expander("API Key Configuration"):
                    st.markdown("""
                    **To enable AI features:**
                    1. Get your OpenAI API key from https://platform.openai.com/api-keys
                    2. Set it as an environment variable: `OPENAI_API_KEY=your-key-here`
                    3. Or add it to Streamlit secrets
                    
                    **Available without API key:**
                    - Basic treatment plan (rule-based)
                    - Case export and management
                    """)
            
            # SINGLE BUTTON FOR COMPLETE WORKFLOW  
            if st.button("Generate Complete AI Analysis & All Documentation", type="primary", help="Analyzes case, generates treatment plan, patient letter, clinical report, and creates PDFs", key="demo_patient_workflow"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Create comprehensive patient data for all processes
                    enhanced_patient_info = patient_info.copy()
                    enhanced_patient_info.update({
                        'symptoms': symptoms,
                        'expectations': expectations,
                        'comorbidities': [c.lower() for c in comorbidities],
                        'age': patient_age,
                        'gender': patient_gender
                    })
                    
                    # Step 1: Generate treatment plan
                    status_text.text("Generating treatment plan...")
                    progress_bar.progress(10)
                    
                    # Generate treatment plan using the planner
                    if 'treatment_planner' in st.session_state:
                        plan = st.session_state.treatment_planner.generate_treatment_plan(prediction, enhanced_patient_info)
                    else:
                        # Fallback if planner not available
                        plan = {
                            "primary_approach": f"Standard care for {prediction['predicted_class']} osteoarthritis",
                            "medications": ["Topical NSAIDs", "Acetaminophen as needed"],
                            "non_pharmacological": ["Physical therapy", "Weight management", "Exercise program"],
                            "lifestyle": ["Low-impact exercises", "Anti-inflammatory diet", "Joint protection"],
                            "follow_up": ["Follow-up in 6-8 weeks", "Monitor symptoms and function"],
                            "red_flags": ["Sudden severe pain", "Signs of infection", "Significant functional decline"]
                        }
                    
                    st.session_state.treatment_plan = plan
                    
                    # Step 2: Generate patient letter
                    status_text.text("Generating personalized patient letter...")
                    progress_bar.progress(25)
                    
                    if api_key_available:
                        patient_letter = generate_patient_letter(enhanced_patient_info, prediction, plan)
                    else:
                        patient_letter = f"""
Dear {patient_info.get('name', 'Patient')},

Your recent knee X-ray has been analyzed using advanced AI technology. The results show **{prediction['predicted_class']} osteoarthritis** with {prediction['confidence']:.1%} confidence.

**What this means:** {prediction['description']}

**Recommended next steps:**
- Follow the treatment plan provided by your healthcare provider
- Schedule a follow-up appointment as recommended
- Contact your doctor if you have questions or concerns

This analysis is intended to support your healthcare team's decision-making. Please discuss these results with your healthcare provider.

Best regards,
Your Healthcare Team
                        """
                    st.session_state.patient_letter = patient_letter
                    
                    # Step 3: Generate exercise plan
                    status_text.text("Creating personalized exercise plan...")
                    progress_bar.progress(40)
                    
                    if api_key_available:
                        exercise_plan = generate_exercise_plan(enhanced_patient_info, prediction['predicted_class'])
                    else:
                        exercise_plan = f"""
**Exercise Recommendations for {prediction['predicted_class']} Osteoarthritis:**

- Low-impact aerobic exercises (walking, swimming, cycling) - 30 minutes, 3-5 times per week
- Quadriceps strengthening exercises - 2-3 times per week
- Range of motion exercises daily
- Balance training to prevent falls
- Avoid high-impact activities that worsen symptoms

**Important:** Consult with your physical therapist before starting any new exercise program.
                        """
                    st.session_state.exercise_plan = exercise_plan
                    
                    # Step 4: Generate lifestyle recommendations
                    status_text.text("Developing lifestyle recommendations...")
                    progress_bar.progress(55)
                    
                    if api_key_available:
                        lifestyle_recommendations = generate_lifestyle_recommendations(enhanced_patient_info, prediction['predicted_class'])
                    else:
                        lifestyle_recommendations = f"""
**Lifestyle Recommendations:**

- **Weight Management:** Maintain healthy weight to reduce joint stress
- **Diet:** Anti-inflammatory foods (fish, vegetables, whole grains)
- **Sleep:** 7-9 hours of quality sleep per night
- **Stress Management:** Practice relaxation techniques
- **Joint Protection:** Use proper body mechanics
- **Heat/Cold Therapy:** Apply as needed for pain relief

**Follow-up:** Schedule regular check-ups with your healthcare provider.
                        """
                    st.session_state.lifestyle_recommendations = lifestyle_recommendations
                    
                    # Step 5: Generate clinical analysis
                    status_text.text("Creating clinical analysis...")
                    progress_bar.progress(70)
                    
                    if api_key_available:
                        clinical_analysis = generate_clinical_analysis(prediction, enhanced_patient_info)
                    else:
                        clinical_analysis = f"""
**Clinical Analysis Summary:**

**AI Prediction:** {prediction['predicted_class']} osteoarthritis
**Confidence Level:** {prediction['confidence']:.1%}
**Patient Age:** {patient_info.get('age', 'Unknown')} years

**Clinical Correlation:** AI analysis suggests {prediction['predicted_class'].lower()} severity. 
Recommend clinical correlation with patient symptoms and physical examination.

**Recommendations:**
- Review AI findings with patient history
- Consider additional imaging if clinically indicated
- Follow evidence-based treatment guidelines
- Monitor treatment response

**Follow-up:** Schedule appropriate follow-up based on severity level.
                        """
                    st.session_state.clinical_analysis = clinical_analysis
                    
                    # Step 6: Generate PDFs
                    status_text.text("Creating PDF documents...")
                    progress_bar.progress(85)
                    
                    if PDF_AVAILABLE:
                        # Generate Patient PDF
                        patient_pdf_bytes = create_patient_pdf(
                            enhanced_patient_info, 
                            prediction, 
                            patient_letter, 
                            exercise_plan, 
                            lifestyle_recommendations
                        )
                        
                        # Generate Clinical PDF
                        clinical_pdf_bytes = create_clinical_pdf(
                            enhanced_patient_info, 
                            prediction, 
                            clinical_analysis, 
                            plan
                        )
                        
                        # Save PDFs to repository
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        patient_filename = f"patient_letter_{patient_info.get('name', 'Patient').replace(' ', '_')}_{timestamp}.pdf"
                        clinical_filename = f"clinical_report_{patient_info.get('name', 'Patient').replace(' ', '_')}_{timestamp}.pdf"
                        
                        # Create new pdf_reports directory structure
                        pdf_reports_dir = REPO_ROOT / "pdf_reports"
                        letters_dir = pdf_reports_dir / "Patient Letters"
                        reports_dir = pdf_reports_dir / "Clinical Reports"
                        letters_dir.mkdir(parents=True, exist_ok=True)
                        reports_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save files
                        with open(letters_dir / patient_filename, 'wb') as f:
                            f.write(patient_pdf_bytes)
                        
                        with open(reports_dir / clinical_filename, 'wb') as f:
                            f.write(clinical_pdf_bytes)
                        
                        # Store in session state for download
                        st.session_state.patient_pdf_bytes = patient_pdf_bytes
                        st.session_state.clinical_pdf_bytes = clinical_pdf_bytes
                        st.session_state.patient_filename = patient_filename
                        st.session_state.clinical_filename = clinical_filename
                        
                        status_text.text("Complete analysis and documentation generated.")
                        progress_bar.progress(100)
                        
                        st.success("Analysis and documentation generated successfully.")
                        st.info(f"Patient letter: `{REPO_NAME}/pdf_reports/Patient Letters/{patient_filename}`")
                        st.info(f"Clinical report: `{REPO_NAME}/pdf_reports/Clinical Reports/{clinical_filename}`")
                        
                        # Show download options
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                            label="Download Patient Letter PDF",
                                data=st.session_state.patient_pdf_bytes,
                                file_name=st.session_state.patient_filename,
                                mime="application/pdf",
                                use_container_width=True
                            )
                        
                        with col2:
                            st.download_button(
                                label="Download Clinical Report PDF", 
                                data=st.session_state.clinical_pdf_bytes,
                                file_name=st.session_state.clinical_filename,
                                mime="application/pdf",
                                use_container_width=True
                            )
                        
                    else:
                        st.error("PDF generation not available. Please install: `pip install reportlab`")
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    status_text.text(f"Error: {str(e)}")
            
            # Show generated content for review
            if any(key in st.session_state for key in ['patient_letter', 'clinical_analysis']):
                st.subheader("Generated Content Review")
                
                # Create tabs for different documents
                doc_tabs = st.tabs(["Patient Letter", "Exercise Plan", "Lifestyle Recommendations", "Clinical Analysis"])
                
                with doc_tabs[0]:
                    if 'patient_letter' in st.session_state:
                        st.markdown("**Patient Letter:**")
                        with st.container():
                            st.write(st.session_state.patient_letter)
                
                with doc_tabs[1]:
                    if 'exercise_plan' in st.session_state:
                        st.markdown("**Exercise Plan:**")
                        with st.container():
                            st.write(st.session_state.exercise_plan)
                
                with doc_tabs[2]:
                    if 'lifestyle_recommendations' in st.session_state:
                        st.markdown("**Lifestyle Recommendations:**")
                        with st.container():
                            st.write(st.session_state.lifestyle_recommendations)
                
                with doc_tabs[3]:
                    if 'clinical_analysis' in st.session_state:
                        st.markdown("**Clinical Analysis:**")
                        with st.container():
                            st.write(st.session_state.clinical_analysis)
            
            st.markdown("---")
    
    # Regular Patient Upload Section
    st.subheader("Upload Patient X-ray")
    
    # Patient info
    st.subheader("Patient Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        patient_age = st.number_input("Age", min_value=18, max_value=120, value=50, key="regular_age")
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="regular_gender")
    with col2:
        symptom_options = [
            "Joint pain", "Stiffness", "Swelling", "Reduced mobility",
            "Grinding sensation", "Joint instability", "Morning stiffness"
        ]
        symptoms = st.multiselect("Current Symptoms", symptom_options, key="regular_symptoms")
    with col3:
        expectation_options = [
            "Pain relief", "Improved mobility", "Prevent progression", "Return to activities", "Surgery avoidance"
        ]
        expectations = st.selectbox("Treatment Expectations", expectation_options, key="regular_expectations")
        comorbidities = st.multiselect("Comorbidities", [
            "Diabetes", "Cardiovascular disease", "Hypertension", "Kidney disease"
        ], key="regular_comorbidities")

    # Image upload
    uploaded_file = st.file_uploader("Upload knee X-ray image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Minimize the upload section once file is uploaded
        st.success(f"File uploaded: {uploaded_file.name}")
        with st.expander("Upload Different Image", expanded=False):
            st.info("To upload a different image, use the file uploader above and select a new file.")
        image = Image.open(uploaded_file)
        c1, c2 = st.columns([1, 2])

        with c1:
            st.image(image, caption="Uploaded X-ray", use_container_width=True)

        with c2:
            with st.spinner("Analyzing X-ray..."):
                prediction = st.session_state.model.predict(image)

            st.markdown('<div class="prediction-box"><h3>🔍 AI Analysis Results</h3></div>', unsafe_allow_html=True)
            severity = prediction["predicted_class"]
            confidence = prediction["confidence"]
            st.metric("Osteoarthritis Severity", severity, delta=f"Confidence: {confidence:.1%}")
            st.write(f"**Description:** {prediction['description']}")

            st.subheader("Probability Distribution")
            prob_data = prediction["all_probabilities"]
            fig = px.bar(x=list(prob_data.keys()), y=list(prob_data.values()),
                        title="Classification Probabilities",
                        labels={"x": "Severity Level", "y": "Probability"})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Treatment plan generated automatically - see streamlined workflow below

# === Treatment Plan Display Functions ===

def display_treatment_plan_with_validation(treatment_plan: Dict, prediction: Dict, actual_severity: str, patient_info: Dict):
    st.markdown('<div class="treatment-section"><h2>Personalized Treatment Plan</h2></div>', unsafe_allow_html=True)
    
    # Show validation status at the top
    is_correct = prediction["predicted_class"] == actual_severity
    if is_correct:
        st.success(f"AI classification correct - Treatment plan based on accurate {prediction['predicted_class']} diagnosis")
    else:
        st.error(f"AI misclassification - Treatment plan based on AI prediction ({prediction['predicted_class']}) but actual severity is {actual_severity}")
        st.warning("**Clinical Note:** Consider this classification discrepancy when evaluating treatment recommendations")
    
    # Check if this is an LLM-generated response or rule-based
    if "llm_response" in treatment_plan:
        # LLM-generated treatment plan
        st.subheader("AI-Generated Treatment Plan")
        st.write(treatment_plan["llm_response"])
        
        # Show additional metadata
        st.caption(f"Generated by: {treatment_plan.get('generated_by', 'AI')} at {treatment_plan.get('timestamp', 'Unknown time')}")
        
        # Add note about LLM usage
        st.info("This treatment plan was generated using advanced AI language models. Please review carefully and consult clinical guidelines.")
        
    else:
        # Rule-based treatment plan
        st.subheader("Primary Treatment Approach")
        st.write(treatment_plan["primary_approach"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Pharmacological Interventions")
            for med in treatment_plan["medications"]:
                    st.write(f"- {med}")
            
            st.subheader("Non-Pharmacological Interventions")
            for item in treatment_plan["non_pharmacological"]:
                    st.write(f"- {item}")
        
        with col2:
            st.subheader("Lifestyle Modifications")
            for item in treatment_plan["lifestyle"]:
                    st.write(f"- {item}")
                
            if treatment_plan.get("surgical_options"):
                st.subheader("Surgical Considerations")
                for opt in treatment_plan["surgical_options"]:
                        st.write(f"- {opt}")

        st.subheader("Follow-up Recommendations")
        for r in treatment_plan["follow_up"]:
            st.write(f"- {r}")

        st.markdown('<div class="warning-box"><h4>Clinical Red Flags - Seek Immediate Attention</h4></div>', unsafe_allow_html=True)
        for flag in treatment_plan["red_flags"]:
            st.write(f"{flag}")

    # Clinical validation section
    st.markdown("---")
    st.subheader("Clinical Validation & Quality Assurance")
    
    col_pred, col_actual, col_match = st.columns(3)
    with col_pred:
        st.metric("AI Prediction", prediction["predicted_class"], delta=f"{prediction['confidence']:.1%} confidence")
    with col_actual:
        st.metric("Actual Classification", actual_severity)
    with col_match:
        match_status = "Correct" if is_correct else "Incorrect"
        if is_correct:
            st.success(f"**{match_status}**")
        else:
            st.error(f"**{match_status}**")
    
    # Detailed clinical analysis
    with st.expander("Detailed Clinical Analysis"):
        st.write(f"**Patient:** {patient_info['name']} ({patient_info['age']}{patient_info['gender'][0]})")
        st.write(f"**Occupation:** {patient_info['occupation']}")
        st.write(f"**AI Prediction:** {prediction['predicted_class']} (Confidence: {prediction['confidence']:.1%})")
        st.write(f"**Ground Truth:** {actual_severity}")
        st.write(f"**Treatment Based On:** AI prediction ({prediction['predicted_class']})")
        
        if is_correct:
            st.success("Validation passed - AI prediction matches expert annotation")
            st.write("**Clinical Implications:**")
            st.write("- Treatment recommendations are based on accurate AI diagnosis")
            st.write("- High confidence in clinical decision support")
            st.write("- No additional validation required")
        else:
            st.error("Validation failed - Misclassification detected")
            st.write("**Clinical Implications:**")
            st.write(f"- AI predicted **{prediction['predicted_class']}** but actual severity is **{actual_severity}**")
            st.write("- Treatment plan may not be optimal for actual condition")
            st.write("- Recommend additional clinical assessment")
            st.write("- Consider expert radiologist review")
            
            # Show what the treatment would be for the actual classification
            st.write("**Alternative Treatment Consideration:**")
            st.write(f"If patient had **{actual_severity}** severity (actual), treatment approach would differ")

    # Clinical references (only for rule-based plans)
    if "references" in treatment_plan:
        with st.expander("Clinical References"):
                for ref in treatment_plan["references"]:
                    st.write(f"- {ref}")

    # === STREAMLINED WORKFLOW: Complete AI Analysis & Documentation ===
    st.subheader("Complete AI Clinical Documentation")
    st.info("Generate all documentation with a single click: treatment plan, patient letter, and clinical report")
    
    # Generate unique case ID if not exists
    if 'case_id' not in st.session_state:
        st.session_state.case_id = f"CASE_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    st.info(f"**Case ID:** {st.session_state.case_id}")
    
    # Check for OpenAI API key
    api_key_available = get_openai_key()
    
    if not api_key_available:
        st.warning("OpenAI API Key required: set OPENAI_API_KEY environment variable to enable AI-generated documentation.")
        with st.expander("API Key Configuration"):
            st.markdown("""
            **To enable AI features:**
            1. Get your OpenAI API key from https://platform.openai.com/api-keys
            2. Set it as an environment variable: `OPENAI_API_KEY=your-key-here`
            3. Or add it to Streamlit secrets
            
            **Available without API key:**
            - Basic treatment plan (rule-based)
            - Case export and management
            """)
    
    # SINGLE BUTTON FOR COMPLETE WORKFLOW
    if st.button("Generate Complete AI Analysis & All Documentation", type="primary", help="Analyzes case, generates treatment plan, patient letter, clinical report, and creates PDFs"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Create comprehensive patient data for all processes
            enhanced_patient_info = patient_info.copy()
            enhanced_patient_info.update({
                'symptoms': patient_data.get('symptoms', []),
                'expectations': patient_data.get('expectations', ''),
                'comorbidities': patient_data.get('comorbidities', [])
            })
            
            # Step 1: Generate treatment plan
            status_text.text("Generating treatment plan...")
            progress_bar.progress(10)
            
            # Generate treatment plan using the planner
            if 'treatment_planner' in st.session_state:
                plan = st.session_state.treatment_planner.generate_treatment_plan(prediction, enhanced_patient_info)
            else:
                # Fallback if planner not available
                plan = {
                    "primary_approach": f"Standard care for {prediction['predicted_class']} osteoarthritis",
                    "medications": ["Topical NSAIDs", "Acetaminophen as needed"],
                    "non_pharmacological": ["Physical therapy", "Weight management", "Exercise program"],
                    "lifestyle": ["Low-impact exercises", "Anti-inflammatory diet", "Joint protection"],
                    "follow_up": ["Follow-up in 6-8 weeks", "Monitor symptoms and function"],
                    "red_flags": ["Sudden severe pain", "Signs of infection", "Significant functional decline"]
                }
            
            st.session_state.treatment_plan = plan
            
            # Step 2: Generate patient letter
            status_text.text("Generating personalized patient letter...")
            progress_bar.progress(25)
            
            if api_key_available:
                patient_letter = generate_patient_letter(enhanced_patient_info, prediction, plan)
            else:
                patient_letter = f"""
Dear {patient_info.get('name', 'Patient')},

Your recent knee X-ray has been analyzed using advanced AI technology. The results show **{prediction['predicted_class']} osteoarthritis** with {prediction['confidence']:.1%} confidence.

**What this means:** {prediction['description']}

**Recommended next steps:**
- Follow the treatment plan provided by your healthcare provider
- Schedule a follow-up appointment as recommended
- Contact your doctor if you have questions or concerns

This analysis is intended to support your healthcare team's decision-making. Please discuss these results with your healthcare provider.

Best regards,
Your Healthcare Team
                """
            st.session_state.patient_letter = patient_letter
            
            # Step 3: Generate exercise plan
            status_text.text("Creating personalized exercise plan...")
            progress_bar.progress(40)
            
            if api_key_available:
                exercise_plan = generate_exercise_plan(enhanced_patient_info, prediction['predicted_class'])
            else:
                exercise_plan = f"""
**Exercise Recommendations for {prediction['predicted_class']} Osteoarthritis:**

- Low-impact aerobic exercises (walking, swimming, cycling) - 30 minutes, 3-5 times per week
- Quadriceps strengthening exercises - 2-3 times per week
- Range of motion exercises daily
- Balance training to prevent falls
- Avoid high-impact activities that worsen symptoms

**Important:** Consult with your physical therapist before starting any new exercise program.
                """
            st.session_state.exercise_plan = exercise_plan
            
            # Step 4: Generate lifestyle recommendations
            status_text.text("Developing lifestyle recommendations...")
            progress_bar.progress(55)
            
            if api_key_available:
                lifestyle_recommendations = generate_lifestyle_recommendations(enhanced_patient_info, prediction['predicted_class'])
            else:
                lifestyle_recommendations = f"""
**Lifestyle Recommendations:**

- **Weight Management:** Maintain healthy weight to reduce joint stress
- **Diet:** Anti-inflammatory foods (fish, vegetables, whole grains)
- **Sleep:** 7-9 hours of quality sleep per night
- **Stress Management:** Practice relaxation techniques
- **Joint Protection:** Use proper body mechanics
- **Heat/Cold Therapy:** Apply as needed for pain relief

**Follow-up:** Schedule regular check-ups with your healthcare provider.
                """
            st.session_state.lifestyle_recommendations = lifestyle_recommendations
            
            # Step 5: Generate clinical analysis
            status_text.text("Creating clinical analysis...")
            progress_bar.progress(70)
            
            if api_key_available:
                clinical_analysis = generate_clinical_analysis(prediction, enhanced_patient_info)
            else:
                clinical_analysis = f"""
**Clinical Analysis Summary:**

**AI Prediction:** {prediction['predicted_class']} osteoarthritis
**Confidence Level:** {prediction['confidence']:.1%}
**Patient Age:** {patient_info.get('age', 'Unknown')} years

**Clinical Correlation:** AI analysis suggests {prediction['predicted_class'].lower()} severity. 
Recommend clinical correlation with patient symptoms and physical examination.

**Recommendations:**
- Review AI findings with patient history
- Consider additional imaging if clinically indicated
- Follow evidence-based treatment guidelines
- Monitor treatment response

**Follow-up:** Schedule appropriate follow-up based on severity level.
                """
            st.session_state.clinical_analysis = clinical_analysis
            
            # Step 6: Generate PDFs
            status_text.text("Creating PDF documents...")
            progress_bar.progress(85)
            
            if PDF_AVAILABLE:
                # Generate Patient PDF
                patient_pdf_bytes = create_patient_pdf(
                    enhanced_patient_info, 
                    prediction, 
                    patient_letter, 
                    exercise_plan, 
                    lifestyle_recommendations
                )
                
                # Generate Clinical PDF
                clinical_pdf_bytes = create_clinical_pdf(
                    enhanced_patient_info, 
                    prediction, 
                    clinical_analysis, 
                    plan
                )
                
                # Save PDFs to repository
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                patient_filename = f"patient_letter_{patient_info.get('name', 'Patient').replace(' ', '_')}_{timestamp}.pdf"
                clinical_filename = f"clinical_report_{patient_info.get('name', 'Patient').replace(' ', '_')}_{timestamp}.pdf"
                
                # Create new pdf_reports directory structure
                pdf_reports_dir = REPO_ROOT / "pdf_reports"
                letters_dir = pdf_reports_dir / "Patient Letters"
                reports_dir = pdf_reports_dir / "Clinical Reports"
                letters_dir.mkdir(parents=True, exist_ok=True)
                reports_dir.mkdir(parents=True, exist_ok=True)
                
                # Save files
                with open(letters_dir / patient_filename, 'wb') as f:
                    f.write(patient_pdf_bytes)
                
                with open(reports_dir / clinical_filename, 'wb') as f:
                    f.write(clinical_pdf_bytes)
                
                # Store in session state for download
                st.session_state.patient_pdf_bytes = patient_pdf_bytes
                st.session_state.clinical_pdf_bytes = clinical_pdf_bytes
                st.session_state.patient_filename = patient_filename
                st.session_state.clinical_filename = clinical_filename
                
                status_text.text("Complete analysis and documentation generated.")
                progress_bar.progress(100)
                
                st.success("Analysis and documentation generated successfully.")
                st.info(f"Patient letter: `{REPO_NAME}/pdf_reports/Patient Letters/{patient_filename}`")
                st.info(f"Clinical report: `{REPO_NAME}/pdf_reports/Clinical Reports/{clinical_filename}`")
                
            else:
                st.error("PDF generation not available. Please install: `pip install reportlab`")
                
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            status_text.text(f"Error: {str(e)}")
    
    # Show download options if PDFs are generated
    if all(key in st.session_state for key in ['patient_pdf_bytes', 'clinical_pdf_bytes']):
        st.subheader("Download Generated Documents")
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
            label="Download Patient Letter PDF",
                data=st.session_state.patient_pdf_bytes,
                file_name=st.session_state.patient_filename,
                mime="application/pdf",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                label="Download Clinical Report PDF", 
                data=st.session_state.clinical_pdf_bytes,
                file_name=st.session_state.clinical_filename,
                mime="application/pdf",
                use_container_width=True
            )
    
    # Show generated content for review
    if any(key in st.session_state for key in ['patient_letter', 'clinical_analysis']):
        st.subheader("Generated Content Review")
        
        # Create tabs for different documents
        doc_tabs = st.tabs(["Patient Letter", "Exercise Plan", "Lifestyle Recommendations", "Clinical Analysis"])
        
        with doc_tabs[0]:
            if 'patient_letter' in st.session_state:
                st.markdown("**Patient Letter:**")
                with st.container():
                    st.write(st.session_state.patient_letter)
        
        with doc_tabs[1]:
            if 'exercise_plan' in st.session_state:
                st.markdown("**Exercise Plan:**")
                with st.container():
                    st.write(st.session_state.exercise_plan)
        
        with doc_tabs[2]:
            if 'lifestyle_recommendations' in st.session_state:
                st.markdown("**Lifestyle Recommendations:**")
                with st.container():
                    st.write(st.session_state.lifestyle_recommendations)
        
        with doc_tabs[3]:
            if 'clinical_analysis' in st.session_state:
                st.markdown("**Clinical Analysis:**")
                with st.container():
                    st.write(st.session_state.clinical_analysis)
    
    # Replace the c1, c2, c3 columns approach with simplified case management
    st.subheader("Case Management")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Generate Patient Letter"):
            try:
                with st.spinner("Generating personalized patient letter..."):
                    # Create comprehensive patient data for LLM
                    enhanced_patient_info = patient_info.copy()
                    enhanced_patient_info.update({
                        'symptoms': patient_data.get('symptoms', []),
                        'expectations': patient_data.get('expectations', ''),
                        'comorbidities': patient_data.get('comorbidities', [])
                    })
                    
                    patient_letter = generate_patient_letter(enhanced_patient_info, prediction, plan)
                    
                    # Generate exercise plan and lifestyle recommendations
                    exercise_plan = generate_exercise_plan(enhanced_patient_info, prediction['predicted_class'])
                    lifestyle_recommendations = generate_lifestyle_recommendations(enhanced_patient_info, prediction['predicted_class'])
                    
                    if PDF_AVAILABLE:
                        pdf_bytes = create_patient_pdf(enhanced_patient_info, prediction, patient_letter, exercise_plan, lifestyle_recommendations)
                        
                        # Create new pdf_reports directory structure
                        pdf_reports_dir = REPO_ROOT / "pdf_reports"
                        letters_dir = pdf_reports_dir / "Patient Letters"
                        letters_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save PDF file
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"patient_letter_{patient_info['name'].replace(' ', '_')}_{timestamp}.pdf"
                        filepath = letters_dir / filename
                        
                        with open(filepath, 'wb') as f:
                            f.write(pdf_bytes)
                        
                        st.success(f"Patient letter generated and saved.")
                        st.info(f"Saved to: `{REPO_NAME}/pdf_reports/Patient Letters/{filename}`")
                        
                        # Offer download
                        st.download_button(
                            label="Download Patient Letter PDF",
                            data=pdf_bytes,
                            file_name=filename,
                            mime="application/pdf"
                        )
                        
                        # Show preview of letter content
                        with st.expander("Preview Patient Letter"):
                            st.write(patient_letter)
                    else:
                        st.error("PDF generation not available. Please install: pip install reportlab")
                        
            except Exception as e:
                st.error(f"Error generating patient letter: {str(e)}")
                
    with c2:
        if st.button("Generate Clinical Report"):
            try:
                with st.spinner("Generating clinical analysis report..."):
                    # Create comprehensive patient data for clinical analysis
                    enhanced_patient_info = patient_info.copy()
                    enhanced_patient_info.update({
                        'symptoms': patient_data.get('symptoms', []),
                        'expectations': patient_data.get('expectations', ''),
                        'comorbidities': patient_data.get('comorbidities', [])
                    })
                    
                    clinical_analysis = generate_clinical_analysis(prediction, enhanced_patient_info)
                    
                    if PDF_AVAILABLE:
                        pdf_bytes = create_clinical_pdf(enhanced_patient_info, prediction, clinical_analysis, plan)
                        
                        # Create new pdf_reports directory structure
                        pdf_reports_dir = REPO_ROOT / "pdf_reports"
                        reports_dir = pdf_reports_dir / "Clinical Reports"
                        reports_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save PDF file
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"clinical_report_{patient_info['name'].replace(' ', '_')}_{timestamp}.pdf"
                        filepath = reports_dir / filename
                        
                        with open(filepath, 'wb') as f:
                            f.write(pdf_bytes)
                        
                        st.success(f"Clinical report generated and saved.")
                        st.info(f"Saved to: `{REPO_NAME}/pdf_reports/Clinical Reports/{filename}`")
                        
                        # Add validation context to the report
                        if is_correct:
                            st.info("**Quality Note:** AI prediction matches ground truth - high confidence in clinical recommendations")
                        else:
                            st.warning("**Quality Note:** AI misclassification detected - manual review recommended")
                        
                        # Offer download
                        st.download_button(
                            label="Download Clinical Report PDF",
                            data=pdf_bytes,
                            file_name=filename,
                            mime="application/pdf"
                        )
                        
                        # Show preview of clinical analysis
                        with st.expander("Preview Clinical Analysis"):
                            st.write(clinical_analysis)
                    else:
                        st.error("PDF generation not available. Please install: pip install reportlab")
                        
            except Exception as e:
                st.error(f"Error generating clinical report: {str(e)}")
                
    with c3:
        if st.button("Export Summary"):
            try:
                # Create summary data
                summary_data = {
                    "patient_name": patient_info['name'],
                    "analysis_date": datetime.datetime.now().isoformat(),
                    "ai_prediction": prediction['predicted_class'],
                    "confidence": prediction['confidence'],
                    "actual_classification": actual_severity,
                    "prediction_correct": is_correct,
                    "patient_context": {
                        "age": patient_info.get('age'),
                        "gender": patient_info.get('gender'),
                        "occupation": patient_info.get('occupation'),
                        "symptoms": patient_data.get('symptoms', []),
                        "comorbidities": patient_data.get('comorbidities', [])
                    },
                    "treatment_plan": plan
                }
                
                # Save as JSON
                reports_dir = REPO_ROOT / "case_summaries"
                reports_dir.mkdir(exist_ok=True)
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"case_summary_{patient_info['name'].replace(' ', '_')}_{timestamp}.json"
                filepath = reports_dir / filename
                
                with open(filepath, 'w') as f:
                    json.dump(summary_data, f, indent=2)
                
                st.success(f"Case summary exported!")
                st.info(f"Saved to: `{REPO_NAME}/case_summaries/{filename}`")
                
                # Offer download
                st.download_button(
                    label="Download Case Summary JSON",
                    data=json.dumps(summary_data, indent=2),
                    file_name=filename,
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"Error exporting summary: {str(e)}")


def display_treatment_plan(treatment_plan: Dict):
    st.markdown('<div class="treatment-section"><h2>Personalized Treatment Plan</h2></div>', unsafe_allow_html=True)

    # Check if this is an LLM-generated response or rule-based
    if "llm_response" in treatment_plan:
        # LLM-generated treatment plan
        st.subheader("AI-Generated Treatment Plan")
        st.write(treatment_plan["llm_response"])
        
        # Show additional metadata
        st.caption(f"Generated by: {treatment_plan.get('generated_by', 'AI')} at {treatment_plan.get('timestamp', 'Unknown time')}")
        
        # Add note about LLM usage
        st.info("This treatment plan was generated using advanced AI language models. Please review carefully and consult clinical guidelines.")
        
    else:
        # Rule-based treatment plan
        st.subheader("Primary Treatment Approach")
        st.write(treatment_plan["primary_approach"])

        st.subheader("Pharmacological Interventions")
        for med in treatment_plan["medications"]:
            st.write(f"- {med}")

        st.subheader("Non-Pharmacological Interventions")
        for item in treatment_plan["non_pharmacological"]:
            st.write(f"- {item}")

        st.subheader("Lifestyle Modifications")
        for item in treatment_plan["lifestyle"]:
            st.write(f"- {item}")

        if treatment_plan.get("surgical_options"):
            st.subheader("Surgical Considerations")
            for opt in treatment_plan["surgical_options"]:
                    st.write(f"- {opt}")

        st.subheader("Follow-up Recommendations")
        for r in treatment_plan["follow_up"]:
            st.write(f"- {r}")

        st.markdown('<div class="warning-box"><h4>Clinical Red Flags - Seek Immediate Attention</h4></div>', unsafe_allow_html=True)
        for flag in treatment_plan["red_flags"]:
            st.write(f"{flag}")

        # Clinical references (only for rule-based plans)
        if "references" in treatment_plan:
            with st.expander("Clinical References"):
                for ref in treatment_plan["references"]:
                    st.write(f"- {ref}")

    st.subheader("Export Options")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Generate Patient Letter", key="regular_patient_letter"):
            try:
                with st.spinner("Generating personalized patient letter..."):
                    # Create patient info for LLM
                    patient_info_for_llm = {
                        'name': 'Patient',
                        'age': patient_data.get('age', 50),
                        'gender': patient_data.get('gender', 'Unknown'),
                        'occupation': 'Not specified',
                        'symptoms': patient_data.get('symptoms', []),
                        'expectations': patient_data.get('expectations', ''),
                        'comorbidities': patient_data.get('comorbidities', []),
                        'bmi': 'Not specified',
                        'activity_level': 'Not specified'
                    }
                    
                    patient_letter = generate_patient_letter(patient_info_for_llm, prediction, plan)
                    
                    # Generate exercise plan and lifestyle recommendations
                    exercise_plan = generate_exercise_plan(patient_info_for_llm, prediction['predicted_class'])
                    lifestyle_recommendations = generate_lifestyle_recommendations(patient_info_for_llm, prediction['predicted_class'])
                    
                    if PDF_AVAILABLE:
                        pdf_bytes = create_patient_pdf(patient_info_for_llm, prediction, patient_letter, exercise_plan, lifestyle_recommendations)
                        
                        # Create new pdf_reports directory structure
                        pdf_reports_dir = REPO_ROOT / "pdf_reports"
                        letters_dir = pdf_reports_dir / "Patient Letters"
                        letters_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save PDF file
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"patient_letter_uploaded_{timestamp}.pdf"
                        filepath = letters_dir / filename
                        
                        with open(filepath, 'wb') as f:
                            f.write(pdf_bytes)
                        
                        st.success(f"Patient letter generated and saved!")
                        st.info(f"Saved to: `{REPO_NAME}/pdf_reports/Patient Letters/{filename}`")
                        
                        # Offer download
                        st.download_button(
                            label="Download Patient Letter PDF",
                            data=pdf_bytes,
                            file_name=filename,
                            mime="application/pdf"
                        )
                        
                        # Show preview of letter content
                        with st.expander("Preview Patient Letter"):
                            st.write(patient_letter)
                    else:
                        st.error("PDF generation not available. Please install: pip install reportlab")
                        
            except Exception as e:
                st.error(f"Error generating patient letter: {str(e)}")
                
    with c2:
        if st.button("Generate Clinical Report", key="regular_clinical_report"):
            try:
                with st.spinner("Generating clinical analysis report..."):
                    # Create patient info for clinical analysis
                    patient_info_for_llm = {
                        'name': 'Patient',
                        'age': patient_data.get('age', 50),
                        'gender': patient_data.get('gender', 'Unknown'),
                        'occupation': 'Not specified',
                        'symptoms': patient_data.get('symptoms', []),
                        'expectations': patient_data.get('expectations', ''),
                        'comorbidities': patient_data.get('comorbidities', []),
                        'bmi': 'Not specified',
                        'activity_level': 'Not specified'
                    }
                    
                    clinical_analysis = generate_clinical_analysis(prediction, patient_info_for_llm)
                    
                    if PDF_AVAILABLE:
                        pdf_bytes = create_clinical_pdf(patient_info_for_llm, prediction, clinical_analysis, plan)
                        
                        # Create new pdf_reports directory structure
                        pdf_reports_dir = REPO_ROOT / "pdf_reports"
                        reports_dir = pdf_reports_dir / "Clinical Reports"
                        reports_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save PDF file
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"clinical_report_uploaded_{timestamp}.pdf"
                        filepath = reports_dir / filename
                        
                        with open(filepath, 'wb') as f:
                            f.write(pdf_bytes)
                        
                        st.success(f"Clinical report generated and saved!")
                        st.info(f"Saved to: `{REPO_NAME}/pdf_reports/Clinical Reports/{filename}`")
                        
                        # Offer download
                        st.download_button(
                            label="Download Clinical Report PDF",
                            data=pdf_bytes,
                            file_name=filename,
                            mime="application/pdf"
                        )
                        
                        # Show preview of clinical analysis
                        with st.expander("Preview Clinical Analysis"):
                            st.write(clinical_analysis)
                    else:
                        st.error("PDF generation not available. Please install: pip install reportlab")
                        
            except Exception as e:
                st.error(f"Error generating clinical report: {str(e)}")
                
    with c3:
        if st.button("Export Summary", key="regular_export_summary"):
            try:
                # Create summary data
                summary_data = {
                    "patient_name": "Uploaded Patient",
                    "analysis_date": datetime.datetime.now().isoformat(),
                    "ai_prediction": prediction['predicted_class'],
                    "confidence": prediction['confidence'],
                    "patient_context": {
                        "age": patient_data.get('age'),
                        "gender": patient_data.get('gender'),
                        "symptoms": patient_data.get('symptoms', []),
                        "comorbidities": patient_data.get('comorbidities', [])
                    },
                    "treatment_plan": plan
                }
                
                # Save as JSON
                reports_dir = REPO_ROOT / "case_summaries"
                reports_dir.mkdir(exist_ok=True)
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"case_summary_uploaded_{timestamp}.json"
                filepath = reports_dir / filename
                
                with open(filepath, 'w') as f:
                    json.dump(summary_data, f, indent=2)
                
                st.success(f"Case summary exported!")
                st.info(f"Saved to: `{REPO_NAME}/case_summaries/{filename}`")
                
                # Offer download
                st.download_button(
                    label="Download Case Summary JSON",
                    data=json.dumps(summary_data, indent=2),
                    file_name=filename,
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"Error exporting summary: {str(e)}")

def batch_processing_interface():
    st.header("Batch Processing")
    
    # Add tabs for different batch processing options
    tab1, tab2 = st.tabs(["Demo Patient Batch", "Upload Multiple Files"])
    
    with tab1:
        st.subheader("Process All Demo Patients")
        demo_patients_dir = REPO_ROOT / "data" / "consensus" / "demo_patients"
        
        if demo_patients_dir.exists():
            # Load demo patient metadata
            metadata_file = demo_patients_dir / "demo_patients_metadata.json"
            demo_metadata = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        demo_metadata = json.load(f)
                except:
                    st.warning("Could not load demo patient metadata")
            
            st.info("Process all 15 demo patients to showcase the full clinical workflow")
            
            if st.button("Analyze All Demo Patients", type="primary"):
                results = []
                progress = st.progress(0)
                status_text = st.empty()
                
                all_demo_files = []
                
                # Collect all demo files
                for severity_dir in sorted(demo_patients_dir.iterdir()):
                    if severity_dir.is_dir():
                        for img_file in severity_dir.glob("*.png"):
                            if img_file.name in demo_metadata:
                                actual_severity = severity_dir.name[1:]  # Remove leading number
                                all_demo_files.append((img_file, demo_metadata[img_file.name], actual_severity))
                
                for i, (img_file, patient_info, actual_severity) in enumerate(all_demo_files):
                    status_text.text(f"Analyzing {patient_info['name']} ({patient_info['age']}{patient_info['gender'][0]})...")
                    
                    try:
                        img = Image.open(img_file)
                        pred = st.session_state.model.predict(img)
                        
                        # Check if prediction matches actual
                        is_correct = pred["predicted_class"] == actual_severity
                        
                        results.append({
                            "patient_name": patient_info['name'],
                            "age": patient_info['age'],
                            "gender": patient_info['gender'],
                            "occupation": patient_info['occupation'],
                            "ai_prediction": pred["predicted_class"],
                            "actual_classification": actual_severity,
                            "prediction_correct": "Correct" if is_correct else "Incorrect",
                            "confidence": pred["confidence"],
                            "activity_level": patient_info.get('activity_level', 'N/A'),
                            "bmi": patient_info.get('bmi', 'N/A'),
                            "comorbidities": ', '.join(patient_info.get('comorbidities', [])) or 'None'
                        })
                    except Exception as e:
                        st.error(f"Error processing {patient_info['name']}: {str(e)}")
                    
                    progress.progress((i + 1) / len(all_demo_files))
                
                status_text.text("Analysis complete!")
                
                if results:
                    df = pd.DataFrame(results)
                    
                    st.subheader("Demo Patient Analysis Results with Validation")
                    st.dataframe(df, use_container_width=True)
                    
                    # Summary statistics with validation metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Patients", len(results))
                    with col2:
                        avg_age = df['age'].mean()
                        st.metric("Average Age", f"{avg_age:.1f} years")
                    with col3:
                        avg_confidence = df['confidence'].mean()
                        st.metric("Average Confidence", f"{avg_confidence:.1%}")
                    with col4:
                        correct_predictions = df['prediction_correct'].str.contains('Correct').sum()
                        accuracy = correct_predictions / len(results)
                        st.metric("AI Accuracy", f"{accuracy:.1%}")
                    
                    # Accuracy analysis
                    st.subheader("Model Performance Analysis")
                    correct_count = df['prediction_correct'].str.contains('Correct').sum()
                    total_count = len(df)
                    
                    st.info(f"**Model Accuracy:** {correct_count}/{total_count} correct predictions ({correct_count/total_count:.1%})")
                    
                    # Show misclassifications
                    incorrect_df = df[df['prediction_correct'].str.contains('Incorrect')]
                    if not incorrect_df.empty:
                        st.subheader("Misclassifications")
                        st.write("Cases where AI prediction differs from ground truth:")
                        misclass_display = incorrect_df[['patient_name', 'ai_prediction', 'actual_classification', 'confidence']].copy()
                        st.dataframe(misclass_display, use_container_width=True)
                    
                    # Accuracy by severity level
                    st.subheader("Accuracy by Severity Level")
                    accuracy_by_severity = df.groupby('actual_classification').agg({
                        'prediction_correct': lambda x: (x.str.contains('Correct')).sum(),
                        'patient_name': 'count'
                    }).rename(columns={'prediction_correct': 'correct_count', 'patient_name': 'total_count'})
                    accuracy_by_severity['accuracy'] = accuracy_by_severity['correct_count'] / accuracy_by_severity['total_count']
                    
                    fig_acc = px.bar(x=accuracy_by_severity.index, y=accuracy_by_severity['accuracy'], 
                                    title="AI Accuracy by Severity Level", 
                                    labels={"x": "Actual Severity", "y": "Accuracy"})
                    fig_acc.update_layout(yaxis=dict(range=[0, 1]))
                    st.plotly_chart(fig_acc, use_container_width=True)
                    
                    # Severity distribution comparison
                    st.subheader("Severity Distribution: AI vs Actual")
                    
                    col_ai, col_actual = st.columns(2)
                    with col_ai:
                        ai_counts = df["ai_prediction"].value_counts()
                        # Create consistent color mapping for severity levels
                        ai_colors = [COLORS['severity'].get(severity, '#636EFA') for severity in ai_counts.index]
                        fig1 = px.pie(values=ai_counts.values, names=ai_counts.index, 
                                     title="AI Predictions", color_discrete_sequence=ai_colors)
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col_actual:
                        actual_counts = df["actual_classification"].value_counts()
                        # Use same color mapping for consistency
                        actual_colors = [COLORS['severity'].get(severity, '#636EFA') for severity in actual_counts.index]
                        fig2 = px.pie(values=actual_counts.values, names=actual_counts.index, 
                                     title="Actual Classifications", color_discrete_sequence=actual_colors)
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Confidence by accuracy
                    st.subheader("Confidence Analysis")
                    df['is_correct'] = df['prediction_correct'].str.contains('Correct')
                    fig3 = px.box(df, x="is_correct", y="confidence", 
                                 title="Prediction Confidence: Correct vs Incorrect")
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # Export option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results with Validation",
                        data=csv,
                        file_name="demo_patient_validation_results.csv",
                        mime="text/csv"
                    )
            
            # === BATCH COMPLETE DOCUMENTATION WORKFLOW ===
            st.subheader("Batch Complete Documentation Generation")
            st.info("Generate treatment plans, patient letters, and clinical reports for ALL demo patients")
            
            # Check for OpenAI API key
            api_key_available = get_openai_key()
            
            if not api_key_available:
                st.warning("⚠️ **OpenAI API Key Required**: Set OPENAI_API_KEY environment variable to enable AI-generated documentation for batch processing.")
                with st.expander("API Key Configuration"):
                    st.markdown("""
                    **To enable batch AI features:**
                    1. Get your OpenAI API key from https://platform.openai.com/api-keys
                    2. Set it as an environment variable: `OPENAI_API_KEY=your-key-here`
                    3. Or add it to Streamlit secrets
                    
                    **Available without API key:**
                    - Basic treatment plans (rule-based) for all patients
                    - Batch case export and management
                    """)
            
            if st.button("Generate Complete Documentation for ALL Demo Patients", type="primary", help="Generates treatment plans, patient letters, clinical reports, and PDFs for all 15 demo patients"):
                batch_progress = st.progress(0)
                batch_status = st.empty()
                
                try:
                    # Initialize batch results storage
                    batch_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    batch_results = []
                    
                    # Ensure treatment planner is available
                    if 'treatment_planner' not in st.session_state:
                        api_key = get_openai_key()
                        st.session_state.treatment_planner = ClinicalTreatmentPlanner(api_key)
                    
                    all_demo_files = []
                    
                    # Collect all demo files again
                    for severity_dir in sorted(demo_patients_dir.iterdir()):
                        if severity_dir.is_dir():
                            for img_file in severity_dir.glob("*.png"):
                                if img_file.name in demo_metadata:
                                    actual_severity = severity_dir.name[1:]  # Remove leading number
                                    all_demo_files.append((img_file, demo_metadata[img_file.name], actual_severity))
                    
                    batch_status.text(f"Starting batch documentation for {len(all_demo_files)} patients...")
                    
                    for i, (img_file, patient_info, actual_severity) in enumerate(all_demo_files):
                        patient_name = patient_info['name']
                        batch_status.text(f"Processing {patient_name} ({i+1}/{len(all_demo_files)})...")
                        
                        try:
                            # Step 1: Get AI prediction
                            img = Image.open(img_file)
                            pred = st.session_state.model.predict(img)
                            
                            # Step 2: Generate treatment plan
                            enhanced_patient_info = patient_info.copy()
                            enhanced_patient_info.update({
                                'symptoms': patient_info.get('symptoms', []),
                                'expectations': patient_info.get('expectations', ''),
                                'comorbidities': patient_info.get('comorbidities', [])
                            })
                            
                            # Generate rule-based treatment plan
                            treatment_plan = st.session_state.treatment_planner.generate_treatment_plan(pred, enhanced_patient_info)
                            
                            # Step 3: Generate patient letter
                            if api_key_available:
                                patient_letter = generate_patient_letter(enhanced_patient_info, pred, treatment_plan)
                            else:
                                patient_letter = f"""
Dear {patient_name},

Your recent knee X-ray has been analyzed using advanced AI technology. The results show **{pred['predicted_class']} osteoarthritis** with {pred['confidence']:.1%} confidence.

**What this means:** {pred['description']}

**Recommended next steps:**
- Follow the treatment plan provided by your healthcare provider
- Schedule a follow-up appointment as recommended
- Contact your doctor if you have questions or concerns

This analysis is intended to support your healthcare team's decision-making. Please discuss these results with your healthcare provider.

Best regards,
Your Healthcare Team
                                """
                            
                            # Step 4: Generate exercise plan
                            if api_key_available:
                                exercise_plan = generate_exercise_plan(enhanced_patient_info, pred['predicted_class'])
                            else:
                                exercise_plan = f"""
**Exercise Recommendations for {pred['predicted_class']} Osteoarthritis:**

• Low-impact aerobic exercises (walking, swimming, cycling) - 30 minutes, 3-5 times per week
• Quadriceps strengthening exercises - 2-3 times per week
• Range of motion exercises daily
• Balance training to prevent falls
• Avoid high-impact activities that worsen symptoms

**Important:** Consult with your physical therapist before starting any new exercise program.
                                """
                            
                            # Step 5: Generate lifestyle recommendations
                            if api_key_available:
                                lifestyle_recommendations = generate_lifestyle_recommendations(enhanced_patient_info, pred['predicted_class'])
                            else:
                                lifestyle_recommendations = f"""
**Lifestyle Recommendations:**

• **Weight Management:** Maintain healthy weight to reduce joint stress
• **Diet:** Anti-inflammatory foods (fish, vegetables, whole grains)
• **Sleep:** 7-9 hours of quality sleep per night
• **Stress Management:** Practice relaxation techniques
• **Joint Protection:** Use proper body mechanics
• **Heat/Cold Therapy:** Apply as needed for pain relief

**Follow-up:** Schedule regular check-ups with your healthcare provider.
                                """
                            
                            # Step 6: Generate clinical analysis
                            if api_key_available:
                                clinical_analysis = generate_clinical_analysis(pred, enhanced_patient_info)
                            else:
                                clinical_analysis = f"""
**Clinical Analysis Summary:**

**AI Prediction:** {pred['predicted_class']} osteoarthritis
**Confidence Level:** {pred['confidence']:.1%}
**Patient Age:** {patient_info.get('age', 'Unknown')} years

**Clinical Correlation:** AI analysis suggests {pred['predicted_class'].lower()} severity. 
Recommend clinical correlation with patient symptoms and physical examination.

**Recommendations:**
- Review AI findings with patient history
- Consider additional imaging if clinically indicated
- Follow evidence-based treatment guidelines
- Monitor treatment response

**Follow-up:** Schedule appropriate follow-up based on severity level.
                                """
                            
                            # Step 7: Generate PDFs
                            if PDF_AVAILABLE:
                                # Generate Patient PDF
                                patient_pdf_bytes = create_patient_pdf(
                                    enhanced_patient_info, 
                                    pred, 
                                    patient_letter, 
                                    exercise_plan, 
                                    lifestyle_recommendations
                                )
                                
                                # Generate Clinical PDF
                                clinical_pdf_bytes = create_clinical_pdf(
                                    enhanced_patient_info, 
                                    pred, 
                                    clinical_analysis, 
                                    treatment_plan
                                )
                                
                                # Save PDFs to batch folders
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                patient_filename = f"patient_letter_{patient_name.replace(' ', '_')}_{timestamp}.pdf"
                                clinical_filename = f"clinical_report_{patient_name.replace(' ', '_')}_{timestamp}.pdf"
                                
                                # Create new pdf_reports batch directories
                                pdf_reports_dir = REPO_ROOT / "pdf_reports"
                                batch_letters_dir = pdf_reports_dir / "Patient Letters" / f"batch_{batch_timestamp}"
                                batch_reports_dir = pdf_reports_dir / "Clinical Reports" / f"batch_{batch_timestamp}"
                                batch_letters_dir.mkdir(parents=True, exist_ok=True)
                                batch_reports_dir.mkdir(parents=True, exist_ok=True)
                                
                                # Save files
                                with open(batch_letters_dir / patient_filename, 'wb') as f:
                                    f.write(patient_pdf_bytes)
                                
                                with open(batch_reports_dir / clinical_filename, 'wb') as f:
                                    f.write(clinical_pdf_bytes)
                                
                                # Store batch result
                                batch_results.append({
                                    "patient_name": patient_name,
                                    "ai_prediction": pred['predicted_class'],
                                    "confidence": pred['confidence'],
                                    "actual_severity": actual_severity,
                                    "patient_pdf": patient_filename,
                                    "clinical_pdf": clinical_filename,
                                    "status": "Complete"
                                })
                            else:
                                batch_results.append({
                                    "patient_name": patient_name,
                                    "ai_prediction": pred['predicted_class'],
                                    "confidence": pred['confidence'],
                                    "actual_severity": actual_severity,
                                    "patient_pdf": "PDF generation unavailable",
                                    "clinical_pdf": "PDF generation unavailable", 
                                    "status": "Analysis only"
                                })
                            
                        except Exception as e:
                            batch_results.append({
                                "patient_name": patient_name,
                                "ai_prediction": "Error",
                                "confidence": 0,
                                "actual_severity": actual_severity,
                                "patient_pdf": "Failed",
                                "clinical_pdf": "Failed",
                                "status": f"Error: {str(e)}"
                            })
                        
                        # Update progress
                        batch_progress.progress((i + 1) / len(all_demo_files))
                    
                    batch_status.text("Batch documentation generation complete!")
                    
                    # Show batch results
                    st.subheader("Batch Documentation Results")
                    batch_df = pd.DataFrame(batch_results)
                    st.dataframe(batch_df, use_container_width=True)
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Processed", len(batch_results))
                    with col2:
                        successful = len([r for r in batch_results if "Complete" in r["status"]])
                        st.metric("Successful", successful)
                    with col3:
                        failed = len([r for r in batch_results if "Error" in r["status"]])
                        st.metric("Failed", failed)
                    with col4:
                        if PDF_AVAILABLE:
                            st.metric("PDFs Generated", successful * 2)
                        else:
                            st.metric("PDFs Generated", 0)
                    
                    if PDF_AVAILABLE and successful > 0:
                        st.success(f"**Batch Documentation Complete!**")
                        st.info(f"**Patient Letters:** `{REPO_NAME}/pdf_reports/Patient Letters/batch_{batch_timestamp}/`")
                        st.info(f"**Clinical Reports:** `{REPO_NAME}/pdf_reports/Clinical Reports/batch_{batch_timestamp}/`")
                    
                    # Export batch summary
                    batch_summary = {
                        "batch_id": f"BATCH_{batch_timestamp}",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "total_patients": len(batch_results),
                        "successful": successful,
                        "failed": failed,
                        "results": batch_results
                    }
                    
                    # Save batch summary
                    batch_dir = REPO_ROOT / "batch_processing"
                    batch_dir.mkdir(exist_ok=True)
                    
                    with open(batch_dir / f"batch_summary_{batch_timestamp}.json", 'w') as f:
                        json.dump(batch_summary, f, indent=2, default=str)
                    
                    # Offer download
                    st.download_button(
                        label="Download Batch Summary",
                        data=json.dumps(batch_summary, indent=2, default=str),
                        file_name=f"batch_summary_{batch_timestamp}.json",
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"Batch processing error: {str(e)}")
                    batch_status.text(f"Error: {str(e)}")
        
        else:
            st.warning("Demo patients directory not found. Please run the data preparation notebook first.")
    
    with tab2:
        st.subheader("Upload Multiple X-ray Images")
        st.info("Upload multiple X-ray images for batch analysis")
        
        uploaded_files = st.file_uploader("Upload multiple X-ray images", 
                                        type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} images")
            if st.button("Process Batch Upload"):
                results = []
                progress = st.progress(0)
                for i, f in enumerate(uploaded_files):
                    img = Image.open(f)
                    pred = st.session_state.model.predict(img)
                    results.append({"filename": f.name, "severity": pred["predicted_class"], 
                                  "confidence": pred["confidence"]})
                    progress.progress((i + 1) / len(uploaded_files))
                
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                st.subheader("Batch Summary")
                counts = df["severity"].value_counts()
                fig = px.pie(values=counts.values, names=counts.index, title="Severity Distribution")
                st.plotly_chart(fig, use_container_width=True)

def patient_dashboard_interface():
    """Comprehensive patient dashboard for viewing, editing, and managing generated reports."""
    st.header("Patient Dashboard")
    st.subheader("Generated Reports Management & Clinical Review")
    
    # Check for generated files
    pdf_reports_dir = REPO_ROOT / "pdf_reports"
    patient_letters_dir = pdf_reports_dir / "Patient Letters"
    clinical_reports_dir = pdf_reports_dir / "Clinical Reports"
    case_summaries_dir = REPO_ROOT / "case_summaries"
    
    # Create main dashboard tabs (consolidated editing into Report Review)
    tab1, tab2 = st.tabs(["Report Review & Editing", "Approval Workflow"])
    
    with tab1:
        st.subheader("Generated Reports Review & Editing")
        
        # File browser
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Report Categories**")
            
            # Count files
            patient_letters = list(patient_letters_dir.glob("*.pdf")) if patient_letters_dir.exists() else []
            clinical_reports = list(clinical_reports_dir.glob("*.pdf")) if clinical_reports_dir.exists() else []
            case_summaries = list(case_summaries_dir.glob("*.json")) if case_summaries_dir.exists() else []
            
            report_type = st.selectbox("Select Report Type", [
                f"Patient Letters ({len(patient_letters)})",
                f"Clinical Reports ({len(clinical_reports)})",
                f"Case Summaries ({len(case_summaries)})"
            ])
            
            # Select specific file
            if "Patient Letters" in report_type and patient_letters:
                selected_file = st.selectbox("Select Patient Letter", [f.name for f in patient_letters])
                if selected_file:
                    file_path = patient_letters_dir / selected_file
                    
            elif "Clinical Reports" in report_type and clinical_reports:
                selected_file = st.selectbox("Select Clinical Report", [f.name for f in clinical_reports])
                if selected_file:
                    file_path = clinical_reports_dir / selected_file
                    
            elif "Case Summaries" in report_type and case_summaries:
                selected_file = st.selectbox("Select Case Summary", [f.name for f in case_summaries])
                if selected_file:
                    file_path = case_summaries_dir / selected_file
            else:
                selected_file = None
                file_path = None
                st.info("No files found in this category")
        
        with col2:
            if file_path and file_path.exists():
                st.markdown(f"**Viewing: {selected_file}**")
                
                if file_path.suffix == '.pdf':
                    # PDF Viewer with download option (simplified for editing focus)
                    st.markdown("**PDF Report Preview**")
                    
                    # Read PDF content
                    with open(file_path, 'rb') as f:
                        pdf_bytes = f.read()
                    
                    # Display PDF information
                    st.info(f"**File:** {selected_file} | **Size:** {len(pdf_bytes):,} bytes")
                    
                    # Simplified display - focus on download since PDFs aren't directly editable
                    try:
                        import base64
                        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                        
                        # Enhanced iframe with better styling
                        pdf_display = f"""
                        <div style="width: 100%; height: 400px; border: 2px solid #ddd; border-radius: 8px; overflow: hidden;">
                            <iframe 
                                src="data:application/pdf;base64,{base64_pdf}" 
                                width="100%" 
                                height="100%" 
                                type="application/pdf"
                                style="border: none;">
                                <p>Your browser does not support PDFs. 
                                <a href="data:application/pdf;base64,{base64_pdf}" download="{selected_file}">Download the PDF</a>.</p>
                            </iframe>
                        </div>
                        """
                        st.markdown(pdf_display, unsafe_allow_html=True)
                        st.caption("PDF preview - Download to edit externally")
                        
                    except Exception as e:
                        st.warning("PDF preview not available. Please download to view.")
                    
                    # Download button
                    st.download_button(
                        label="Download PDF",
                        data=pdf_bytes,
                        file_name=selected_file,
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                    st.info("**Note:** PDF files can be downloaded and edited externally. Case Summaries can be edited directly below.")
                    
                elif file_path.suffix == '.json':
                    # JSON Case Summary Viewer & Editor (INTEGRATED EDITING)
                    st.markdown("**Case Summary - View & Edit**")
                    
                    try:
                        with open(file_path, 'r') as f:
                            case_data = json.load(f)
                        
                        # Display key information
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Patient", case_data.get('patient_name', 'Unknown'))
                            st.metric("AI Prediction", case_data.get('ai_prediction', 'Unknown'))
                        with col_b:
                            st.metric("Confidence", f"{case_data.get('confidence', 0):.1%}")
                            is_correct = case_data.get('prediction_correct')
                            if is_correct is not None:
                                st.metric("Validation", "Correct" if is_correct else "Incorrect")
                        
                        # === INTEGRATED EDITING FUNCTIONALITY ===
                        st.markdown("---")
                        st.markdown("### **Edit Case Information**")
                        
                        # Extract original content for editing
                        patient_info = case_data.get('patient_context', {})
                        
                        # Edit patient information
                        st.markdown("**👤 Patient Information**")
                        col1, col2 = st.columns(2)
                        with col1:
                            edited_age = st.number_input("Age", value=patient_info.get('age', 50), min_value=18, max_value=120, key="edit_age")
                            edited_gender = st.selectbox("Gender", ["Male", "Female", "Other"], 
                                                       index=["Male", "Female", "Other"].index(patient_info.get('gender', 'Other')) 
                                                       if patient_info.get('gender') in ["Male", "Female", "Other"] else 0, key="edit_gender")
                        with col2:
                            edited_occupation = st.text_input("Occupation", value=patient_info.get('occupation', ''), key="edit_occupation")
                            edited_symptoms = st.multiselect("Symptoms", 
                                                           ["Joint pain", "Stiffness", "Swelling", "Reduced mobility", "Grinding sensation"],
                                                           default=patient_info.get('symptoms', []), key="edit_symptoms")
                        
                        # Treatment plan editing
                        treatment_plan = case_data.get('treatment_plan', {})
                        if isinstance(treatment_plan, dict):
                            st.markdown("**Treatment Plan**")
                            edited_primary = st.text_area("Primary Approach", 
                                                         value=treatment_plan.get('primary_approach', ''),
                                                         height=100, key="edit_primary")
                            edited_medications = st.text_area("Medications", 
                                                            value='\n'.join(treatment_plan.get('medications', [])),
                                                            height=100, key="edit_medications")
                            edited_lifestyle = st.text_area("Lifestyle Modifications", 
                                                           value='\n'.join(treatment_plan.get('lifestyle', [])),
                                                           height=100, key="edit_lifestyle")
                        
                        # === AI-POWERED EDITING SECTION ===
                        st.markdown("---")
                        st.markdown("### **AI-Powered Editing Assistant**")
                        
                        if not OPENAI_AVAILABLE or not get_openai_key():
                            st.warning("⚠️ OpenAI API key not configured. AI-powered editing features are not available.")
                            st.info("Configure your API key in the Settings tab to enable AI assistance.")
                        else:
                            # AI editing options
                            col_ai1, col_ai2 = st.columns(2)
                            with col_ai1:
                                ai_action = st.selectbox("Select AI Action", [
                                    "Improve Patient Letter",
                                    "Enhance Clinical Analysis", 
                                    "Suggest Treatment Modifications",
                                    "Generate Follow-up Recommendations",
                                    "Create Patient Education Content"
                                ], key="ai_action")
                            
                            with col_ai2:
                                # Custom instructions
                                custom_instructions = st.text_area(
                                    "Additional Instructions for AI",
                                    placeholder="E.g., 'Make the language more accessible for elderly patients' or 'Focus on preventive care measures'",
                                    height=100,
                                    key="ai_instructions"
                                )
                            
                            if st.button("Generate AI Suggestions", key="ai_generate"):
                                with st.spinner(f"AI generating {ai_action.lower()}..."):
                                    try:
                                        context = f"Case: {case_data.get('patient_name', 'Unknown')}\nAI Prediction: {case_data.get('ai_prediction', 'Unknown')}\nConfidence: {case_data.get('confidence', 0):.1%}"
                                        
                                        # Generate AI suggestions
                                        suggestions = generate_ai_editing_suggestions(ai_action, context, custom_instructions)
                                        
                                        st.markdown("**AI Suggestions:**")
                                        st.write(suggestions)
                                        
                                    except Exception as e:
                                        st.error(f"AI suggestion error: {str(e)}")
                        
                        # === SAVE CHANGES SECTION ===
                        st.markdown("---")
                        col_save1, col_save2 = st.columns(2)
                        
                        with col_save1:
                            if st.button("Save Changes", type="primary", key="save_changes"):
                                # Update case data
                                case_data['patient_context'].update({
                                    'age': edited_age,
                                    'gender': edited_gender,
                                    'occupation': edited_occupation,
                                    'symptoms': edited_symptoms
                                })
                                
                                if isinstance(treatment_plan, dict):
                                    case_data['treatment_plan'].update({
                                        'primary_approach': edited_primary,
                                        'medications': [m.strip() for m in edited_medications.split('\n') if m.strip()],
                                        'lifestyle': [l.strip() for l in edited_lifestyle.split('\n') if l.strip()]
                                    })
                                
                                case_data['last_modified'] = datetime.datetime.now().isoformat()
                                case_data['modified_by'] = 'Medical Professional Edit'
                                
                                # Save updated case
                                with open(file_path, 'w') as f:
                                    json.dump(case_data, f, indent=2)
                                
                                st.success("Changes saved successfully!")
                                st.rerun()
                        
                        with col_save2:
                            # Show original data for reference
                            with st.expander("View Original Case Data"):
                                st.json(case_data)
                            
                    except Exception as e:
                        st.error(f"Error reading case summary: {str(e)}")
                        
                # File metadata
                stat = file_path.stat()
                st.caption(f"Created: {datetime.datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')}")
                st.caption(f"Size: {stat.st_size:,} bytes")
                
            else:
                st.info("Select a Case Summary from the Report Review tab to edit")
                
                # Quick stats
                if patient_letters or clinical_reports or case_summaries:
                    st.markdown("**Quick Stats**")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Patient Letters", len(patient_letters))
                    with col_b:
                        st.metric("Clinical Reports", len(clinical_reports))
                    with col_c:
                        st.metric("Case Summaries", len(case_summaries))
                    
                    st.info("**Medical Doctor Instructions:** Select a Case Summary to edit reports using LLM prompts or direct text editing.")
                else:
                    st.warning("No reports found. Generate some reports using the Single Patient Analysis first!")
    
    with tab2:
        st.subheader("Clinical Approval Workflow")
        
        # Approval status management
        if file_path and file_path.exists() and file_path.suffix == '.json':
            try:
                with open(file_path, 'r') as f:
                    case_data = json.load(f)
                
                st.markdown(f"**Review Status for: {case_data.get('patient_name', 'Unknown')}**")
                
                # Current status
                current_status = case_data.get('approval_status', 'pending_review')
                reviewed_by = case_data.get('reviewed_by', 'Not reviewed')
                review_date = case_data.get('review_date', 'Not reviewed')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Status", current_status.replace('_', ' ').title())
                with col2:
                    st.metric("Reviewed By", reviewed_by)
                with col3:
                    st.metric("Review Date", review_date[:10] if review_date != 'Not reviewed' else review_date)
                
                # Quality metrics
                st.markdown("**Quality Assessment**")
                prediction_correct = case_data.get('prediction_correct')
                if prediction_correct is not None:
                    if prediction_correct:
                        st.success("AI Prediction Validated - Matches Ground Truth")
                    else:
                        st.warning("AI Misclassification Detected - Requires Review")
                
                confidence = case_data.get('confidence', 0)
                confidence_threshold = st.session_state.get('confidence_threshold', 0.7)
                
                if confidence >= confidence_threshold:
                    st.success(f"High Confidence ({confidence:.1%}) - Above threshold ({confidence_threshold:.1%})")
                else:
                    st.warning(f"Low Confidence ({confidence:.1%}) - Below threshold ({confidence_threshold:.1%})")
                
                # Review actions
                st.markdown("**Clinical Review Actions**")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    reviewer_name = st.text_input("Reviewer Name", placeholder="Dr. Smith")
                    new_status = st.selectbox("Update Status", [
                        "pending_review",
                        "under_review", 
                        "approved",
                        "needs_revision",
                        "rejected"
                    ], index=["pending_review", "under_review", "approved", "needs_revision", "rejected"].index(current_status))
                
                with col_b:
                    review_notes = st.text_area("Review Notes", 
                                              placeholder="Clinical observations, concerns, or approval notes...",
                                              height=120)
                    
                    priority_level = st.selectbox("Priority Level", ["Low", "Medium", "High", "Urgent"])
                
                # Update approval status
                if st.button("Update Review Status"):
                    case_data.update({
                        'approval_status': new_status,
                        'reviewed_by': reviewer_name,
                        'review_date': datetime.datetime.now().isoformat(),
                        'review_notes': review_notes,
                        'priority_level': priority_level
                    })
                    
                    with open(file_path, 'w') as f:
                        json.dump(case_data, f, indent=2)
                    
                    st.success(f"Review status updated to: {new_status.replace('_', ' ').title()}")
                    st.rerun()
                
                # Review history
                if case_data.get('review_history'):
                    with st.expander("Review History"):
                        for i, review in enumerate(case_data['review_history']):
                            st.write(f"**Review {i+1}:** {review.get('status', 'Unknown')} by {review.get('reviewer', 'Unknown')} on {review.get('date', 'Unknown')}")
                            if review.get('notes'):
                                st.write(f"Notes: {review['notes']}")
                            st.divider()
                
            except Exception as e:
                st.error(f"Error loading approval data: {str(e)}")
        else:
            st.info("Select a Case Summary to manage approval workflow")
            
            # Workflow overview
            st.markdown("**Approval Workflow Overview**")
            st.write("1. **Pending Review** - New cases awaiting clinical review")
            st.write("2. **Under Review** - Cases currently being evaluated")
            st.write("3. **Approved** - Cases cleared for clinical use")
            st.write("4. **Needs Revision** - Cases requiring modifications")
            st.write("5. **Rejected** - Cases not suitable for clinical use")
    
    # Quick actions sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Quick Actions**")
        
        if st.button("Refresh Dashboard"):
            st.rerun()
        
        if st.button("Export All Data"):
            try:
                # Create comprehensive export
                export_data = {
                    'export_date': datetime.datetime.now().isoformat(),
                    'patient_letters': len(patient_letters) if 'patient_letters' in locals() else 0,
                    'clinical_reports': len(clinical_reports) if 'clinical_reports' in locals() else 0,
                    'case_summaries': len(case_summaries) if 'case_summaries' in locals() else 0,
                    'dashboard_version': '1.0'
                }
                
                export_json = json.dumps(export_data, indent=2)
                st.download_button(
                    label="Download Export",
                    data=export_json,
                    file_name=f"dashboard_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"Export failed: {str(e)}")

def generate_ai_editing_suggestions(action: str, context: str, instructions: str) -> str:
    """Generate AI-powered editing suggestions for medical content."""
    if not OPENAI_AVAILABLE or not get_openai_key():
        return "AI editing not available - OpenAI API key not configured."
    
    openai.api_key = get_openai_key()
    
    action_prompts = {
        "Improve Patient Letter": "Enhance this patient letter to be more empathetic, accessible, and encouraging while maintaining medical accuracy.",
        "Enhance Clinical Analysis": "Improve this clinical analysis with additional evidence-based insights and recommendations.",
        "Suggest Treatment Modifications": "Suggest personalized treatment plan modifications based on current best practices.",
        "Generate Follow-up Recommendations": "Create comprehensive follow-up and monitoring recommendations.",
        "Create Patient Education Content": "Generate educational content to help patients understand their condition and treatment."
    }
    
    prompt = f"""
You are an expert medical communication specialist and clinician. 

Context: {context}

Task: {action_prompts.get(action, action)}

Additional Instructions: {instructions if instructions else 'None provided'}

Please provide specific, actionable suggestions that are:
1. Medically accurate and evidence-based
2. Appropriate for the intended audience (patient vs. clinician)
3. Compassionate and empowering
4. Practical and implementable

Format your response as clear, numbered recommendations.
"""

    try:
        return chat_completion(
            messages=[
                {"role": "system", "content": "You are an expert medical editor and communication specialist."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3,
            model="gpt-4o-mini",
        ).strip()
    except Exception as e:
        return f"Error generating AI suggestions: {str(e)}"

def analytics_dashboard():
    st.header("Analytics Dashboard")
    
    # Check if we have generated reports to analyze
    pdf_reports_dir = REPO_ROOT / "pdf_reports"
    patient_letters_dir = pdf_reports_dir / "Patient Letters"
    clinical_reports_dir = pdf_reports_dir / "Clinical Reports"
    case_summaries_dir = REPO_ROOT / "case_summaries"
    
    # Create tabs for different analytics views
    tab1, tab2, tab3, tab4 = st.tabs(["Case Analytics", "Model Performance", "Report Management", "Future Features"])
    
    with tab1:
        st.subheader("Generated Cases Analytics")
        
        # Check for case summaries
        if case_summaries_dir.exists():
            case_files = list(case_summaries_dir.glob("*.json"))
            if case_files:
                st.success(f"Found {len(case_files)} analyzed cases")
                
                # Load and analyze case data
                cases_data = []
                for case_file in case_files:
                    try:
                        with open(case_file, 'r') as f:
                            case_data = json.load(f)
                            cases_data.append(case_data)
                    except:
                        continue
                
                if cases_data:
                    # Create summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Cases", len(cases_data))
                    with col2:
                        severities = [case.get('ai_prediction', 'Unknown') for case in cases_data]
                        most_common = max(set(severities), key=severities.count) if severities else 'N/A'
                        st.metric("Most Common Severity", most_common)
                    with col3:
                        confidences = [case.get('confidence', 0) for case in cases_data if case.get('confidence')]
                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                    with col4:
                        correct_predictions = [case for case in cases_data if case.get('prediction_correct', False)]
                        accuracy = len(correct_predictions) / len(cases_data) if cases_data else 0
                        st.metric("Accuracy (Demo Patients)", f"{accuracy:.1%}")
                    
                    # Severity distribution chart
                    severity_counts = {}
                    for case in cases_data:
                        severity = case.get('ai_prediction', 'Unknown')
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1
                    
                    if severity_counts:
                        fig = px.pie(values=list(severity_counts.values()), 
                                   names=list(severity_counts.keys()),
                                   title="Severity Distribution of Analyzed Cases")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Recent cases table
                    st.subheader("Recent Cases")
                    recent_cases = sorted(cases_data, key=lambda x: x.get('analysis_date', ''), reverse=True)[:10]
                    
                    if recent_cases:
                        df_recent = pd.DataFrame([
                            {
                                'Patient': case.get('patient_name', 'Unknown'),
                                'Date': case.get('analysis_date', '')[:10] if case.get('analysis_date') else 'Unknown',
                                'AI Prediction': case.get('ai_prediction', 'Unknown'),
                                'Confidence': f"{case.get('confidence', 0):.1%}" if case.get('confidence') else 'N/A',
                                'Correct': 'Correct' if case.get('prediction_correct') else 'Incorrect' if case.get('prediction_correct') is False else '?'
                            }
                            for case in recent_cases
                        ])
                        st.dataframe(df_recent, use_container_width=True)
                else:
                    st.info("No valid case data found in summary files")
            else:
                st.info("No case summaries found. Analyze some patients first to see analytics.")
        else:
            st.info("No case summaries directory found. Generate some reports first!")
    
    with tab2:
        st.subheader("Model Performance Metrics")
        st.info("**Planned Features:**")
        st.write("- Model accuracy tracking over time")
        st.write("- Confidence distribution analysis") 
        st.write("- Performance by severity level")
        st.write("- False positive/negative analysis")
        st.write("- Model calibration metrics")
        
        # Show basic confidence threshold usage
        if 'confidence_threshold' in st.session_state:
            threshold = st.session_state.confidence_threshold
            st.subheader("Current Confidence Settings")
            st.write(f"**Active Threshold:** {threshold:.1%}")
            st.write(f"**Effect:** Predictions below {threshold:.1%} confidence will be flagged for manual review")
    
    with tab3:
        st.subheader("Generated Reports Management")
        
        # Count generated files
        patient_letters_count = len(list(patient_letters_dir.glob("*.pdf"))) if patient_letters_dir.exists() else 0
        clinical_reports_count = len(list(clinical_reports_dir.glob("*.pdf"))) if clinical_reports_dir.exists() else 0
        case_summaries_count = len(list(case_summaries_dir.glob("*.json"))) if case_summaries_dir.exists() else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Patient Letters", patient_letters_count)
        with col2:
            st.metric("Clinical Reports", clinical_reports_count)
        with col3:
            st.metric("Case Summaries", case_summaries_count)
        
        if patient_letters_count > 0 or clinical_reports_count > 0:
            st.success("Generated files are saved in the repository:")
            if patient_letters_count > 0:
                st.write(f"• **Patient Letters:** `{REPO_NAME}/pdf_reports/Patient Letters/` ({patient_letters_count} files)")
            if clinical_reports_count > 0:
                st.write(f"• **Clinical Reports:** `{REPO_NAME}/pdf_reports/Clinical Reports/` ({clinical_reports_count} files)")
            if case_summaries_count > 0:
                st.write(f"• **Case Summaries:** `{REPO_NAME}/case_summaries/` ({case_summaries_count} files)")
                
            st.info("**Future Enhancement:** Patient dashboard for viewing, editing, and managing generated PDFs")
        else:
            st.info("No reports generated yet. Use the Single Patient Analysis to generate your first reports!")
    
    with tab4:
        st.subheader("Planned Analytics Features")
        
        st.markdown("""
        ### **Short-term Goals (Next Update)**
        - **Patient Dashboard**: View and edit generated PDFs
        - **LLM Integration Analytics**: Track AI recommendation quality
        - **Batch Analysis Results**: Comprehensive batch processing reports
        - **Export/Import**: Backup and restore case data
        
        ### **Medium-term Goals**
        - **Outcome Tracking**: Follow-up patient progress
        - **Treatment Effectiveness**: Analyze treatment plan success rates
        - **Model Performance Monitoring**: Real-time accuracy tracking
        - **Clinical Decision Support**: Evidence-based recommendation scoring
        
        ### **Long-term Vision**
        - **Multi-modal Analysis**: Integrate clinical notes, lab results
        - **Predictive Analytics**: Risk stratification and outcome prediction
        - **Quality Assurance**: Automated quality checks for AI predictions
        - **Research Integration**: Export data for clinical research
        
        ### **Technical Implementation Notes**
        - **Data Storage**: JSON-based case summaries for lightweight analytics
        - **PDF Management**: ReportLab-generated reports with versioning
        - **LLM Integration**: GPT-4 powered personalized recommendations
        - **Validation**: Ground truth comparison for demo patients
        """)
        
        st.info("**Contributing**: This is an open-source medical AI project. Analytics features can be expanded based on clinical needs and user feedback.")

def settings_interface():
    st.header("Settings")
    
    st.subheader("AI Model Configuration")
    model_status = "Loaded" if "model" in st.session_state else "Not Loaded"
    st.info(f"Model Status: {model_status}")
    
    if "model" in st.session_state:
        # Check if using demo model or trained model
        model_info = "Demo ResNet-50 model" if hasattr(st.session_state.model, '_using_demo') else "Trained ensemble model"
        st.write(f"**Active Model:** {model_info}")
        
        # Model path information
        expected_path = REPO_ROOT / "models" / "deployment" / "best_model_for_deployment.pth"
        if expected_path.exists():
            st.success(f"Trained model found: `{REPO_NAME}/models/deployment/best_model_for_deployment.pth`")
        else:
            st.warning(f"Trained model not found at: `{REPO_NAME}/models/deployment/best_model_for_deployment.pth`")
            st.info("**To get the trained model:** Run notebook `04_Multi_Class_Full_Training_Ensemble.ipynb` to train the ensemble model.")
    
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.7, 
        step=0.05,
        help="Minimum AI confidence level to highlight high-confidence predictions. Predictions below this threshold will be flagged for additional review."
    )
    
    # Save threshold to session state for use across the app
    st.session_state.confidence_threshold = confidence_threshold
    
    if confidence_threshold < 0.5:
        st.warning("⚠️ Very low confidence threshold - most predictions will be flagged for review")
    elif confidence_threshold > 0.9:
        st.info("ℹHigh confidence threshold - only very confident predictions will be auto-approved")
    
    st.subheader("LLM Integration Setup")
    
    # Check current API key status
    current_api_key = get_openai_key()
    secrets_file = Path(".streamlit/secrets.toml")
    template_file = Path(".streamlit/secrets.toml.template")
    
    if current_api_key:
        st.success("OpenAI API key configured - AI treatment recommendations enabled!")
        
        # Test LLM functionality
        st.markdown("**Test LLM Integration**")
        st.caption("This tests whether the OpenAI API connection is working for generating AI-powered treatment recommendations and patient letters.")
        
        if st.button("Run LLM Test"):
            try:
                test_classification = {
                    "predicted_class": "Mild",
                    "confidence": 0.87,
                    "description": "Mild osteoarthritis with minor joint changes"
                }
                test_patient = {
                    "age": 55, "gender": "Female", "symptoms": ["Joint pain"], 
                    "expectations": "Pain relief", "comorbidities": []
                }
                
                with st.spinner("Testing OpenAI API connection and treatment plan generation..."):
                    plan = st.session_state.treatment_planner.generate_treatment_plan(test_classification, test_patient)
                
                if plan.get("generated_by") == "LLM":
                    st.success("LLM integration test successful! AI-powered recommendations are working.")
                    st.info("The app can now generate personalized patient letters and advanced treatment recommendations using GPT-4.")
                else:
                    st.info("ℹLLM test completed - using rule-based fallback (this is normal without API key)")
                    st.warning("⚠️ Without LLM integration, the app uses predefined treatment templates instead of personalized AI recommendations.")
                    
            except Exception as e:
                st.error(f"LLM test failed: {str(e)}")
                st.info("This usually means the API key is invalid or OpenAI service is unavailable.")
        
        # Show API key status (masked)
        masked_key = current_api_key[:8] + "..." + current_api_key[-4:] if len(current_api_key) > 12 else "***"
        st.code(f"API Key: {masked_key}")
        
        st.markdown("**Refresh LLM Connection**")
        st.caption("Use this if you've updated your API key or want to reinitialize the LLM connection.")
        
        if st.button("Refresh Integration"):
            st.session_state.treatment_planner = ClinicalTreatmentPlanner(current_api_key)
            st.success("LLM integration refreshed! New API key settings applied.")
            st.info("You can now test the connection again with the updated settings.")
            
    else:
        st.warning("OpenAI API key not configured")
        st.info("**LLM-powered treatment recommendations are currently disabled.** The app will use rule-based recommendations instead.")
        
        # Setup instructions
        st.markdown("### **Setup Instructions for AI Treatment Recommendations**")
        
        # Check if template exists
        if template_file.exists():
            st.markdown("**Step 1:** Set up your API key configuration")
            
            if not secrets_file.exists():
                st.code(f"cp .streamlit/secrets.toml.template .streamlit/secrets.toml")
                st.markdown("Or copy the template file manually:")
                st.markdown(f"Copy `.streamlit/secrets.toml.template` → `.streamlit/secrets.toml`")
            else:
                st.info("Secrets file already exists: `.streamlit/secrets.toml`")
            
            st.markdown("**Step 2:** Add your OpenAI API key")
            st.markdown("1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)")
            st.markdown("2. Edit `.streamlit/secrets.toml` and replace `your-openai-api-key-here` with your actual API key")
            st.markdown("3. Restart the application")
            
            # Show template content
            with st.expander("View secrets.toml template"):
                try:
                    with open(template_file, 'r') as f:
                        template_content = f.read()
                    st.code(template_content, language="toml")
                except:
                    st.error("Could not read template file")
        else:
            st.error("Template file not found: `.streamlit/secrets.toml.template`")
            st.markdown("Create `.streamlit/secrets.toml` manually with:")
            st.code('OPENAI_API_KEY = "your-api-key-here"', language="toml")
        
        # Manual override option
        with st.expander("**Temporary API Key Override** (for testing)"):
            st.warning("⚠️ This method is less secure - use only for testing")
            api_key = st.text_input("Temporary OpenAI API Key", type="password", 
                                   help="Enter your OpenAI API key for this session only")
            if api_key:
                if api_key.startswith("sk-"):
                    st.session_state.treatment_planner = ClinicalTreatmentPlanner(api_key)
                    st.success("LLM integration enabled for this session")
                    st.info("For permanent setup, use the secrets.toml method above")
                else:
                    st.error("Invalid API key format. OpenAI keys should start with 'sk-'")
    
    # System Information
    st.subheader("System Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Repository:** {REPO_NAME}")
        st.info(f"**PyTorch:** {torch.__version__}")
        st.info(f"**Device:** {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    with col2:
        if torch.cuda.is_available():
            st.info(f"**GPU:** {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            st.info(f"**GPU Memory:** {memory_gb:.1f} GB")
        else:
            st.info("**GPU:** Not available")
        
        st.info(f"**OpenAI Available:** {'Yes' if OPENAI_AVAILABLE else 'No'}")
    
    # Demo Patients Status
    st.subheader("Demo Patients Status")
    demo_dir = REPO_ROOT / "data" / "consensus" / "demo_patients"
    metadata_file = demo_dir / "demo_patients_metadata.json"
    
    if demo_dir.exists() and metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                demo_metadata = json.load(f)
            
            st.success(f"Demo patients ready: {len(demo_metadata)} patients loaded")
            
            # Quick stats
            if demo_metadata:
                ages = [p.get('age', 0) for p in demo_metadata.values()]
                genders = [p.get('gender', 'Unknown') for p in demo_metadata.values()]
                st.info(f"Age range: {min(ages)}-{max(ages)} years | Gender: {len([g for g in genders if g == 'Female'])}F, {len([g for g in genders if g == 'Male'])}M")
                
        except Exception as e:
            st.error(f"Error loading demo patient metadata: {str(e)}")
    else:
        st.warning("Demo patients not found")
        st.info("**To set up demo patients:** Run notebook `01_Data_Preparation.ipynb` to create demo patient profiles")

if __name__ == "__main__":
    main() 