
# Crop Care AI
### Intelligent Plant Disease Detection & Doctor System

**Crop Care AI** is a comprehensive agricultural assistant designed to bridge the gap between identifying plant diseases and finding the right treatment. By combining computer vision with a smart chatbot, this system acts as both a **Pathologist** (to detect the disease) and a **Doctor** (to prescribe the cure).

---

## About The Project

In agriculture, timing is everything. Farmers often struggle not just with identifying what is wrong with their crops, but also with knowing the exact medicine or organic cure to apply. Traditional AI tools often stop at detection, leaving the farmer to guess the solution.

We built **Crop Care AI** to solve this. It uses a **Hybrid Deep Learning model** (CNN + Transformer) to analyze leaf images with high accuracy. Once a disease is found, our **RAG-based Chatbot** steps in to provide verified medicines, preventive measures, and treatment guides instantly.

---

## Key Features

* **Accurate Disease Detection:** Capable of identifying **38 different plant diseases** across 14 crop species (including Tomato, Potato, Apple, Corn, and Grape).
* **The "Gatekeeper" System:** A smart filtering layer (using MobileNetV3) that checks if an uploaded image is actually a plant. If a user uploads a shoe or a car, the system blocks it to prevent false results.
* **AI Plant Doctor:** A specialized chatbot that uses **Retrieval-Augmented Generation (RAG)**. Unlike standard AI that might "guess" answers, this system retrieves verified cures and medicines from a trusted database.
* **Real-Time Analysis:** Built on Streamlit for a fast, responsive, and user-friendly web interface.

---

##  Tech Stack

* **Frontend:** Streamlit (Python)
* **Deep Learning:** PyTorch, Torchvision
* **Model Architecture:**
    * *Vision:* Hybrid ResNet50 (Feature Extraction) + Transformer Encoder (Global Context)
    * *Safety:* MobileNetV3 (Gatekeeper)
* **Natural Language Processing:** Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`)
* **Data Processing:** Pandas, NumPy, Pickle

---
## How It Works (System Logic)

The system follows a strict four-step pipeline to ensure accuracy and reliability:

1.  **Input:**
    The user uploads a high-resolution photo of a plant leaf via the web interface.

2.  **Safety Check (The Gatekeeper):**
    Before processing, a lightweight **MobileNetV3** model scans the image.
    * *Pass:* If it detects a plant/crop, the image moves to the next stage.
    * *Block:* If it detects a non-plant object (e.g., a shoe, car, or animal), the system halts and alerts the user to upload a valid image.

3.  **🔍 Disease Detection (Hybrid Model):**
    The validated image is processed by our dual-engine architecture:
    * **ResNet50 (The Eye):** Extracts local visual features like textures, edges, and spot colors.
    * **Transformer Encoder (The Brain):** Applies *Self-Attention* to analyze the global pattern and spread of the disease across the leaf.
    * *Result:* The model predicts the specific disease (e.g., *"Tomato Early Blight"*) with a confidence score.

4.  **Advisory (RAG Chatbot):**
    The detected disease name is automatically sent to the AI Doctor module.
    * The chatbot searches its vectorized knowledge base for the specific **medicine, chemical cure, and organic remedy** associated with that disease.
    * The actionable advice is presented immediately to the user without needing manual searching.
##  Screenshots

### 1. Home Page
*The landing page providing an overview of the project features.*
![Home Page](<img width="1470" height="956" alt="Screenshot 2025-12-08 at 11 06 06 AM" src="https://github.com/user-attachments/assets/540c71da-dce5-4770-8699-537a144b1b3c" />
)

### 2. Disease Detection Result
*The AI successfully identifying the plant disease and providing a confidence score.*
![Disease Detection](<img width="348" height="478" alt="Screenshot 2025-12-06 at 7 02 08 PM" src="https://github.com/user-attachments/assets/a7676c93-04b1-445d-9e0f-8a235e113d82" />
)

### 3. Plant Doctor Chatbot
*The RAG-based chatbot recommending specific cures and medicines for the detected disease.*
![Plant Doctor Chatbot](<img width="1470" height="956" alt="Screenshot 2025-12-03 at 9 00 36 PM" src="https://github.com/user-attachments/assets/e0832727-0fda-40a8-8f77-ade2115556e2" />
)

