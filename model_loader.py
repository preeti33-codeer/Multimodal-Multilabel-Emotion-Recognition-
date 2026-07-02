from transformers import pipeline
import torch

DEVICE = 0 if torch.cuda.is_available() else -1

text_pipe = None
image_pipe = None
audio_pipe = None


def get_text_model():
    global text_pipe

    if text_pipe is None:
        print("Loading Text Emotion Model...")
        text_pipe = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=DEVICE
        )
        print("✓ Text Model Ready")

    return text_pipe


def get_image_model():
    global image_pipe

    if image_pipe is None:
        print("Loading Image Emotion Model...")
        image_pipe = pipeline(
            "image-classification",
            model="trpakov/vit-face-expression",
            top_k=5,
            device=DEVICE
        )
        print("✓ Image Model Ready")

    return image_pipe


def get_audio_model():
    global audio_pipe

    if audio_pipe is None:
        print("Loading Audio Emotion Model...")
        audio_pipe = pipeline(
            "audio-classification",
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            top_k=6,
            device=DEVICE
        )
        print("✓ Audio Model Ready")

    return audio_pipe