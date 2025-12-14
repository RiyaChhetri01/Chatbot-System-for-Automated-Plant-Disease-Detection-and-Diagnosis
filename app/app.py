import os
import sys

# --- 1. CRITICAL CONFIGURATION (Must be top) ---
# Prevents Mac freezing/deadlock issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from collections import OrderedDict
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import requests

# --- IMPORT SENTENCE_TRANSFORMERS ---
from sentence_transformers import SentenceTransformer

# --- 2. PAGE CONFIG ---
st.set_page_config(
    page_title="Crop Care AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #f4f9f4; }
    h1 { color: #2e7d32; font-family: 'Helvetica Neue', sans-serif; }
    h2, h3 { color: #1b5e20; }
    div.stButton > button {
        background-color: #2e7d32;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    div.stButton > button:hover { background-color: #1b5e20; color: white; }
    [data-testid="stSidebar"] { background-color: #e8f5e9; border-right: 1px solid #c8e6c9; }
    </style>
""", unsafe_allow_html=True)

# --- 4. MODEL CLASSES (Vision) ---
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-2])
    def forward(self, x): return self.feature_extractor(x)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim=2048, heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dim))
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, x):
        attn, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn)
        x = self.norm2(x + self.ffn(x))
        return x

class CNN_Transformer_Model(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.transformer = TransformerEncoderBlock(dim=2048)
        self.fc = nn.Linear(2048, num_classes)
    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten(2).permute(2, 0, 1)
        x = self.transformer(x)
        x = self.fc(x.mean(dim=0))
        return x

# --- 5. CONSTANTS & HELPER ---
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___healthy", "Cherry___Powdery_mildew",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust", "Corn___healthy", "Corn___Northern_Leaf_Blight",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___healthy",
    "Potato___Late_blight", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___healthy", "Strawberry___Leaf_scorch", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___healthy", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# --- 6. LOADING LOGIC ---
@st.cache_resource
def load_models():
    # A. Vision Model
    vision_model = CNN_Transformer_Model(num_classes=38)
    vision_path = os.path.join(os.getcwd(), "model", "cnn_transformer_model.pth")
    if os.path.exists(vision_path):
        state = torch.load(vision_path, map_location=torch.device('cpu'))
        new_state = OrderedDict()
        for k, v in state.items(): new_state[k.replace("module.", "")] = v
        vision_model.load_state_dict(new_state, strict=False)
        vision_model.eval()
    else:
        vision_model = None

    # B. Gatekeeper Model (MobileNet)
    # We include this to prevent non-plant images (like shoes) from giving false results.
    try:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        gate_model = models.mobilenet_v3_small(weights=weights)
        gate_model.eval()
        gate_prep = weights.transforms()
        # Load ImageNet labels for the Gatekeeper
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        gate_classes = requests.get(url).text.splitlines()
    except:
        gate_model, gate_prep, gate_classes = None, None, None

    # C. Chatbot Model
    if os.path.exists('./chatbot_model'):
        chat_model = SentenceTransformer('./chatbot_model', device='cpu')
    else:
        chat_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
    
    try:
        with open('./pickleFiles/faq_embeddings.pkl', 'rb') as f: emb = pickle.load(f)
        with open('./pickleFiles/chatbot_data.pkl', 'rb') as f: df = pickle.load(f)
    except:
        emb, df = None, None

    return vision_model, gate_model, gate_prep, gate_classes, chat_model, emb, df

# --- 7. INITIALIZE ---
with st.spinner("🌱 Starting Crop Care AI..."):
    vision_model, gate_model, gate_prep, gate_classes, chat_model, faq_embeddings, df = load_models()

# --- 8. SIDEBAR ---
# --- 8. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=80)
    st.title("Crop Care AI")
    st.markdown("---")
    
    # CRITICAL CHANGE: Added key="navigation" so we can change it automatically
    page = st.radio("Navigate", ["🏠 Home", "📸 Disease Detector", "💬 Plant Doctor"], key="navigation")
    
    
    

# --- 9. MAIN PAGES ---

# === HOME ===
if page == "🏠 Home":
    st.title("🌿 Welcome to Crop Care AI")
    st.markdown("### Intelligent Plant Disease Diagnosis & Treatment")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("This AI-powered system helps you:")
        st.markdown("- **Detect** diseases from leaf photos.")
        st.markdown("- **Diagnose** issues with high accuracy.")
        st.markdown("- **Consult** an AI Doctor for cures.")
        st.success("👈 Click **Disease Detector** to begin.")
    with col2:
        # Changed use_column_width to use_container_width
        st.image("https://cdn.dribbble.com/users/2063381/screenshots/15446700/media/894347781a9667749065096576f3f03b.png", use_container_width=True)

# === DETECTOR ===
elif page == "📸 Disease Detector":
    st.title("📸 AI Disease Diagnosis")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Upload Leaf")
        uploaded_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            # Changed use_column_width to use_container_width
            st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("2. Results")
        if uploaded_file and vision_model:
            if st.button("🔍 Analyze Leaf", use_container_width=True):
                with st.spinner("Analyzing..."):
                    try:
                        # --- 1. GATEKEEPER CHECK (Improved) ---
                        if gate_model and gate_classes:
                            gate_in = gate_prep(image).unsqueeze(0)
                            with torch.no_grad(): gate_out = gate_model(gate_in)
                            gate_prob = torch.nn.functional.softmax(gate_out, dim=1)
                            detected_obj = gate_classes[torch.argmax(gate_prob).item()]
                            
                            # EXPANDED KEYWORD LIST to prevent false blocking
                            plant_keywords = [
                                'leaf', 'plant', 'flower', 'fruit', 'vegetable', 'crop', 'corn', 
                                'apple', 'grape', 'agriculture', 'cabbage', 'broccoli', 'pot', 
                                'greenhouse', 'fungus', 'mushroom', 'tree', 'grass', 'lemon', 
                                'orange', 'banana', 'pomegranate', 'produce', 'food', 'head cabbage'
                            ]
                            is_plant = any(k in detected_obj.lower() for k in plant_keywords)
                            
                            if not is_plant:
                                st.error(f"**Invalid Image Detected**")
                                st.warning(f"This looks like a **{detected_obj}**, not a plant leaf.")
                                st.stop() # Stops execution here if not a plant

                        # --- 2. MAIN DISEASE PREDICTION ---
                        input_tensor = preprocess_image(image)
                        with torch.no_grad(): output = vision_model(input_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1)
                        top_prob, top_idx = torch.topk(probs, 1)
                        confidence = top_prob.item()
                        disease_name = CLASS_NAMES[top_idx.item()].replace("___", " - ").replace("_", " ")

                        if confidence < 0.85:
                            st.warning(f"⚠️ Low Confidence ({confidence*100:.1f}%).")
                        else:
                            st.success(f"**Detected:** {disease_name}")
                            st.metric("Confidence", f"{confidence*100:.2f}%")
                            
                            # --- 3. ASK DOCTOR BUTTON (INTEGRATION) ---
                            # --- 3. ASK DOCTOR BUTTON (INTEGRATION) ---
                            st.markdown("### 🩺 Next Steps")
                            
                            # We use a callback to change the page immediately
                            def go_to_chat():
                                st.session_state['auto_ask'] = f"How to cure {disease_name}?"
                                st.session_state['navigation'] = "💬 Plant Doctor" # Must match the Radio option exactly
                            
                            st.button(
                                f"Chat with Doctor about this", 
                                use_container_width=True, 
                                on_click=go_to_chat
                            )

                    except Exception as e:
                        st.error(f"Error: {e}")

# === CHATBOT ===
elif page == "💬 Plant Doctor":
    st.title("💬 Plant Doctor")
    
    if chat_model:
        chat_container = st.container()
        
        # Initialize history
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AI Plant Doctor. Ask me about any plant disease."}]
        
        # Display history
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        
        # Check for Auto-Ask from Detector
        auto_prompt = st.session_state.pop('auto_ask', None)
        user_input = st.chat_input("Ex: How to cure Potato Early Blight?")
        
        # Determine prompt source
        final_prompt = auto_prompt if auto_prompt else user_input
        
        if final_prompt:
            # Display user message (only if manual input, auto-ask usually jumps straight to answer)
            if not auto_prompt:
                with chat_container:
                    st.chat_message("user").markdown(final_prompt)
                    st.session_state.messages.append({"role": "user", "content": final_prompt})
            
            # Logic
            vec = chat_model.encode([final_prompt])
            sims = cosine_similarity(vec, faq_embeddings)
            best_idx = np.argmax(sims)
            score = sims[0][best_idx].item()
            
            if score > 0.4:
                response = df.iloc[best_idx]['response']
            else:
                response = "I'm not sure. Please ask specifically about a plant disease."
            
            with chat_container:
                st.chat_message("assistant").markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})