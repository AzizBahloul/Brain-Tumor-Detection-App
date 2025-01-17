import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('brain_tumor_model.keras')

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
        st.image(img, caption='Uploaded Image.', use_column_width=True)
    
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