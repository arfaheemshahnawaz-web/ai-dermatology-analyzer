# gradio_derm_chatbot_interactive.py
import cv2
import numpy as np
from PIL import Image
import gradio as gr
from groq import Groq
import os

# -----------------------------
#  HARDCODED GROQ API KEY
# -----------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")   # <<< put your key here

# Global to keep metrics for the current uploaded image (used across follow-up messages)
METRICS = None

# ---------- simple image heuristics ----------
def analyze_image_cv(np_img):
    img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    total_pixels = h * w

    r, g, b = img[:,:,0].astype(int), img[:,:,1].astype(int), img[:,:,2].astype(int)
    red_mask = (r > 120) & (r > g + 30) & (r > b + 30)
    redness_pct = float(red_mask.sum()) / total_pixels * 100

    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = h * w
    lesion_contours = [c for c in contours if cv2.contourArea(c) > img_area * 0.0005]

    brightness = float(np.mean(gray))

    return {
        "width": w,
        "height": h,
        "redness_pct": round(redness_pct, 2),
        "lesion_count_est": len(lesion_contours),
        "brightness": round(brightness, 1)
    }

# ---------- build prompts ----------
def build_initial_prompt(metrics):
    return f"""
You are a dermatologist giving general, non-diagnostic guidance based on IMAGE METRICS:
- Size: {metrics['width']}x{metrics['height']}
- Redness: {metrics['redness_pct']}%
- Estimated lesion count: {metrics['lesion_count_est']}
- Brightness: {metrics['brightness']}

TASK:
1) Give a cautious assessment of likely skin type and acne type (inflammatory vs comedonal) using cautious language.
2) Recommend a safe, evidence-based OTC skincare routine (short, stepwise).
3) List red-flag symptoms requiring in-person dermatologist evaluation.
4) End with a medical disclaimer.
"""

def build_followup_prompt(metrics, user_question, recent_chat):
    # recent_chat is a list of past [user,bot] pairs so we can optionally summarise or include it.
    # Keep it short — include metrics and the user's question for context.
    return f"""
You are a dermatologist assistant. Use the following image-derived metrics to answer the user's question.

IMAGE METRICS:
- Size: {metrics['width']}x{metrics['height']}
- Redness: {metrics['redness_pct']}%
- Estimated lesion count: {metrics['lesion_count_est']}
- Brightness: {metrics['brightness']}

RECENT CHAT (most recent entries):
{(recent_chat[-6:] if recent_chat else [])}

USER QUESTION:
{user_question}

Answer concisely and clearly. Remind user it's general advice and not a diagnosis if relevant.
"""

# ---------- LLM call wrapper ----------
def call_groq(prompt, max_tokens=700):
    if not GROQ_API_KEY:
        return "No API key provided in code. Please add your Groq key to GROQ_API_KEY."
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

# ---------- Handlers ----------
def analyze_image(image, chat_history):
    global METRICS
    if image is None:
        return chat_history

    # convert PIL -> OpenCV BGR
    img = np.array(image.convert("RGB"))[:, :, ::-1].copy()
    metrics = analyze_image_cv(img)
    METRICS = metrics

    prompt = build_initial_prompt(metrics)
    answer = call_groq(prompt)

    chat_history = chat_history or []
    chat_history.append(["User: uploaded image", answer])
    return chat_history

def user_ask(user_message, chat_history):
    global METRICS
    chat_history = chat_history or []
    if not METRICS:
        chat_history.append(["User: " + user_message, "Please upload an image first so I can answer using the picture's context."])
        return "", chat_history

    prompt = build_followup_prompt(METRICS, user_message, chat_history)
    answer = call_groq(prompt)
    # append the exchange
    chat_history.append([f"User: {user_message}", answer])
    # return cleared input (""), updated chat history
    return "", chat_history

# ---------- UI ----------
with gr.Blocks() as demo:
    gr.Markdown(
        "# DermAI\n\n"
        "1) Upload a clear photo of the affected area and click **Analyze Image**.\n"
        "2) Ask follow-up questions in the chat box (e.g., \"What topical should I use?\").\n\n"
        "**Important:** This app provides general advice only and is **not** a medical diagnosis."
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(label="Upload skin image (well-lit)", type="pil")
            analyze_btn = gr.Button("Analyze Image")
            user_input = gr.Textbox(placeholder="Type a follow-up question...", label="Ask the bot")
            send_btn = gr.Button("Send")
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(label="Dermatology Guidance")

    # Analyze image -> update chatbot only
    analyze_btn.click(analyze_image, inputs=[image_in, chatbot], outputs=[chatbot])
    # Send message -> clear textbox and update chatbot
    send_btn.click(user_ask, inputs=[user_input, chatbot], outputs=[user_input, chatbot])
    # allow Enter key to submit
    user_input.submit(user_ask, inputs=[user_input, chatbot], outputs=[user_input, chatbot])

if __name__ == "__main__":
    demo.launch()
