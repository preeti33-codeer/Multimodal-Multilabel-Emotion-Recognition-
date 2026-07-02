# === Multimodal Emotion Recognition (Flexible Modalities) ===
# Supports analyzing only one, two, or all three modalities (Text / Image / Audio)

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import time

from PIL import Image

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from predict import (
    analyze_text,
    analyze_image,
    analyze_audio
)

import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# CONFIG
# -------------------------------
CSV_FILE = "emotion_results.csv"

# Ensure CSV
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=[
        "timestamp","text","text_label","text_score",
        "image_label","image_score","audio_label","audio_score"
    ]).to_csv(CSV_FILE, index=False)

# -------------------------------
# Load Models (same models as original demo)
# -------------------------------

# -------------------------------
# Helpers
# -------------------------------



def save_result(row):
    df = pd.read_csv(CSV_FILE)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)


def create_chart(scores_map):
    # scores_map: {'Text':(label,score), 'Image':(...), 'Audio':(...)} for selected modalities
    labels = list(scores_map.keys())
    scores = [scores_map[k][1] for k in labels]
    emotions = [scores_map[k][0] for k in labels]

    fig, ax = plt.subplots(figsize=(6,3))
    bars = ax.bar(labels, scores)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence")
    ax.set_title("Emotion Confidence per Modality")

    for b, emo, sc in zip(bars, emotions, scores):
        height = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, height / 2, f"{emo}\n{sc:.2f}",
                ha="center", va="center", fontsize=9, color='white')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# -------------------------------
# Train simple ML models (optional)
# -------------------------------

def train_models():
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        return None, None

    df = pd.read_csv(CSV_FILE)
    if df.empty: return None, None

    # If older rows have N/A or missing labels, we still try
    def combine_emotions(row):
        labels = [str(row.get("text_label","N/A")), str(row.get("image_label","N/A")), str(row.get("audio_label","N/A"))]
        labels = [l for l in labels if l and l != "N/A"]
        if not labels: return "Neutral"
        return max(set(labels), key=labels.count)

    df["overall_emotion"] = df.apply(combine_emotions, axis=1)
    X = df[["text_score", "image_score", "audio_score"]].fillna(0)
    y = df["overall_emotion"]

    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=3)

    tree.fit(X, y)
    knn.fit(X, y)
    return tree, knn

# -------------------------------
# Main: flexible processing based on selected modalities
# -------------------------------

def process_modalities(user_text, user_image, user_audio, selected_modalities):
    # selected_modalities: list like ['Text','Image'] etc.
    if not selected_modalities:
        return " Select at least one modality to analyze.", None, None, None

    # Validate provided inputs for chosen modalities
    if 'Text' in selected_modalities and (not user_text or str(user_text).strip() == ""):
        return " Text modality selected but no text provided.", None, None, None
    if 'Image' in selected_modalities and user_image is None:
        return " Image modality selected but no image provided.", None, None, None
    if 'Audio' in selected_modalities and user_audio is None:
        return " Audio modality selected but no audio provided.", None, None, None

    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    # Default outputs
    t_label, t_score = "N/A", 0.0
    i_label, i_score = "N/A", 0.0
    a_label, a_score = "N/A", 0.0

    # Analyze only selected
    if 'Text' in selected_modalities:
        t_label, t_score = analyze_text(user_text)
    if 'Image' in selected_modalities:
        i_label, i_score = analyze_image(user_image)
    if 'Audio' in selected_modalities:
        a_label, a_score = analyze_audio(user_audio)

    # Prepare save row — keep N/A for unselected
    row = {
        "timestamp": ts,
        "text": user_text if 'Text' in selected_modalities else "",
        "text_label": t_label,
        "text_score": round(t_score, 3),
        "image_label": i_label,
        "image_score": round(i_score, 3),
        "audio_label": a_label,
        "audio_score": round(a_score, 3)
    }
    save_result(row)

    # Chart only for selected modalities
    scores_map = {}
    if 'Text' in selected_modalities: scores_map['Text'] = (t_label, t_score)
    if 'Image' in selected_modalities: scores_map['Image'] = (i_label, i_score)
    if 'Audio' in selected_modalities: scores_map['Audio'] = (a_label, a_score)

    chart = create_chart(scores_map)

    # Train and predict (if enough data)
    tree, knn = train_models()
    dt_pred, knn_pred = "N/A", "N/A"
    if tree and knn:
        sample = [[t_score, i_score, a_score]]
        dt_pred = tree.predict(sample)[0]
        knn_pred = knn.predict(sample)[0]

    # Build dynamic summary that only shows selected modalities
    lines = [f"###  Multimodal Emotion Results\n🕒 **Time:** {ts}\n"]
    if 'Text' in selected_modalities:
        lines.append(f"**Text:** {t_label} ({t_score:.2f})")
    if 'Image' in selected_modalities:
        lines.append(f"**Image:** {i_label} ({i_score:.2f})")
    if 'Audio' in selected_modalities:
        lines.append(f"**Audio:** {a_label} ({a_score:.2f})")

    lines.append('\n---\n')
    lines.append(f"###  Decision Tree Predicted: **{dt_pred}**")
    lines.append(f"### KNN Predicted: **{knn_pred}**\n")
    lines.append(f" Saved to `{CSV_FILE}`")

    summary = "\n".join(lines)

    df = pd.read_csv(CSV_FILE).tail(10).reset_index(drop=True)
    return summary, chart, CSV_FILE, df

# -------------------------------
# Gradio UI
# -------------------------------

title = " Multi Model Multi Label Emotion Recognition — Select modalities to analyze"
desc = "Choose one, two or all three modalities and provide only the inputs for those you selected."

css = """
.gradio-container {background: linear-gradient(to right, #ffecd2 0%, #fcb69f 100%);}
h1,h3 {color:#333;}
.gr-button {background-color:#ff8c00;color:white;border-radius:20px;}
.gr-button:hover {background-color:#e07b00;}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(desc)

    with gr.Row():
        with gr.Column(scale=5):
            with gr.Group():
                gr.Markdown("## Inputs")
                modalities = gr.CheckboxGroup(choices=["Text","Image","Audio"], value=["Text","Image","Audio"], label="Select modalities to analyze")
                txt = gr.Textbox(label=" Enter your feeling", placeholder="Type how you feel...", lines=2)
                img = gr.Image(label=" Capture or Upload Photo", sources=["webcam", "upload"], type="pil")
                aud = gr.Audio(label=" Record your voice", sources=["microphone"], type="filepath")
                btn = gr.Button(" Analyze Emotions", variant="primary")

        with gr.Column(scale=5):
            with gr.Group():
                gr.Markdown("## Results")
                out_md = gr.Markdown("Awaiting input...")
                out_chart = gr.Image(label="📊 Confidence Chart")
                out_csv = gr.File(label="⬇️ Download Results CSV")
                out_df = gr.Dataframe(label="Recent Predictions", interactive=False)

    btn.click(process_modalities, inputs=[txt, img, aud, modalities], outputs=[out_md, out_chart, out_csv, out_df])

# Run demo
if __name__ == '__main__':
    demo.launch(share=True)