# DermAI – AI Dermatology Assistant

DermAI is an AI-powered dermatology assistant that analyzes skin images and provides general skincare guidance using computer vision and large language models.

The system combines **OpenCV image analysis** with an **LLM chatbot powered by Groq** to give users contextual dermatological insights and follow-up advice.

⚠️ This application is for educational purposes only and should not replace professional medical diagnosis.

---

# Features

- Upload a skin image for analysis
- Computer vision-based image metrics (redness, lesion estimation, brightness)
- AI-powered dermatology guidance
- Interactive chatbot for follow-up questions
- Gradio web interface

---

# How It Works

1. User uploads a skin image.
2. OpenCV analyzes the image and extracts metrics:
   - redness percentage
   - estimated lesion count
   - brightness
3. Metrics are passed to an AI model (Groq LLM).
4. The AI provides general dermatological insights.
5. Users can ask follow-up questions through the chatbot.

---

# Tech Stack

### Frontend
- Gradio

### Backend
- Python

### Computer Vision
- OpenCV
- NumPy
- Pillow

### AI Model
- Groq LLM (LLaMA 3)

---

# Installation

```bash
Clone the repository:
git clone https://github.com/arfaheemshahnawaz-web/ai-dermatology-analyzer
