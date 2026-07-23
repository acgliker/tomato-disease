import os
import requests
import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "tomato_resnet18.pth"
CONFIDENCE_THRESHOLD = 0.70

advice_dict = {
    "Tomato___Bacterial_spot": "Possible bacterial spot. Remove infected leaves, avoid overhead watering, and improve air circulation.",
    "Tomato___Early_blight": "Possible early blight. Remove affected leaves, keep foliage dry, and consider using a suitable fungicide.",
    "Tomato___Late_blight": "Possible late blight. Isolate the plant, remove infected parts, and avoid wet leaves. Severe cases may need disposal.",
    "Tomato___Leaf_Mold": "Possible leaf mold. Improve ventilation, reduce humidity, and avoid watering the leaves directly.",
    "Tomato___Septoria_leaf_spot": "Possible Septoria leaf spot. Remove infected lower leaves and prevent soil from splashing onto leaves.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Possible spider mite damage. Check the underside of leaves, rinse the plant, and consider insecticidal soap.",
    "Tomato___Target_Spot": "Possible target spot. Remove affected leaves, improve airflow, and avoid excessive leaf moisture.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Possible yellow leaf curl virus. Check for whiteflies and remove severely infected plants.",
    "Tomato___Tomato_mosaic_virus": "Possible tomato mosaic virus. Avoid touching healthy plants after infected ones and remove infected plant material.",
    "Tomato___healthy": "The leaf appears healthy. Continue regular care with proper light, watering, and ventilation.",
}


def get_openrouter_key():
    try:
        return st.secrets.get("OPENROUTER_API_KEY", "")
    except Exception:
        return os.getenv("OPENROUTER_API_KEY", "")


@st.cache_resource
def load_model():
    device = "cpu"

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location=device)

    classes = checkpoint["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, classes, device


def get_ai_advice(predicted_class, confidence, fixed_advice):
    api_key = get_openrouter_key()

    if not api_key:
        return None, "No OpenRouter API key was found."

    prompt = (
        f"A tomato leaf disease classifier predicted: {predicted_class}. "
        f"Confidence: {confidence:.2%}. "
        f"Built-in advice: {fixed_advice} "
        "Give short, practical care advice for a home grower. "
        "Use simple English. Mention this is not a professional diagnosis."
    )

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://streamlit.app",
                "X-Title": "Tomato Disease Recognition App",
            },
            json={
                "model": "openai/gpt-oss-20b:free",
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )

        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"], None

    except Exception as error:
        return None, str(error)


st.title("Tomato Leaf Disease Recognition")
st.write("Upload a tomato leaf image to identify possible disease.")

model, classes, device = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

uploaded_file = st.file_uploader(
    "Choose a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    result = classes[predicted.item()]
    confidence_value = confidence.item()

    st.subheader("Prediction Result")
    st.write(f"Class: **{result}**")
    st.write(f"Confidence: **{confidence_value:.2%}**")

    if confidence_value < CONFIDENCE_THRESHOLD:
        st.warning(
            "The confidence is low, so this result may not be reliable. "
            "Please try a clearer tomato leaf image or check the plant manually."
        )

    fixed_advice = advice_dict.get(
        result,
        "No built-in care advice is available for this class."
    )

    st.subheader("Built-in Care Advice")
    st.write(fixed_advice)

    st.subheader("AI Enhanced Advice")

    if st.button("Generate AI Advice"):
        with st.spinner("Generating AI advice..."):
            ai_advice, error = get_ai_advice(result, confidence_value, fixed_advice)

        if ai_advice:
            st.write(ai_advice)
        else:
            st.info("AI advice is unavailable, so the app is using built-in care advice.")
            st.caption(f"Reason: {error}")