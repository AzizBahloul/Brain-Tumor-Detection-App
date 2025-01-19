import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Define the model architecture
def build_model():
    base_model = EfficientNetB7(
        weights=None,  # We will load the weights separately
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    
    return model

# Load the model architecture
model = build_model()

# Load the model weights
model.load_weights('model.weights.h5')

# Define class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.sidebar.header('Brain Tumor Detection')
st.sidebar.info('Upload an MRI image to detect the type of brain tumor.')

st.title('🧠 Brain Tumor Detection App')

uploaded_file = st.file_uploader("📥 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        img = image.load_img(uploaded_file, target_size=(224, 224))
        st.image(img, caption='Uploaded Image.', use_container_width=True)
    
    with col2:
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model.predict(img_array)
        confidence = np.max(prediction) * 100
        tumor_type = class_names[np.argmax(prediction)]
        
        st.success(f"**Predicted Tumor:** {tumor_type}")
        st.info(f"**Confidence:** {confidence:.2f}%")
        
        st.write("**Prediction Probabilities:**")
        prob_df = {class_names[i]: f"{prediction[0][i]*100:.2f}%" for i in range(len(class_names))}
        st.bar_chart(list(prediction[0]))
        
        st.write(prob_df)