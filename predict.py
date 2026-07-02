from model_loader import (
    get_text_model,
    get_image_model,
    get_audio_model
)


def analyze_text(text):
    try:
        model = get_text_model()

        result = model(text)

        if isinstance(result, list):
            result = result[0]

        return (
            str(result["label"]).title(),
            float(result["score"])
        )

    except Exception as e:
        print("Text Error:", e)
        return "Neutral", 0.0


def analyze_image(image):
    try:
        model = get_image_model()

        result = model(image)

        if isinstance(result, list):
            result = result[0]

        return (
            str(result["label"]).title(),
            float(result["score"])
        )

    except Exception as e:
        print("Image Error:", e)
        return "Neutral", 0.0


def analyze_audio(audio_path):
    try:
        model = get_audio_model()

        result = model(audio_path)

        if isinstance(result, list):
            result = result[0]

        return (
            str(result["label"]).title(),
            float(result["score"])
        )

    except Exception as e:
        print("Audio Error:", e)
        return "Neutral", 0.0