Brain Tumor Detection Using Deep Learning 🧠
Welcome to the Brain Tumor Detection project, a cutting-edge deep learning application designed to detect and classify brain tumors from MRI scans with high accuracy. Powered by a modified VGG16 convolutional neural network (CNN), this tool classifies brain scans into four categories: Glioma, Meningioma, Pituitary Tumor, and No Tumor. This repository provides a robust, user-friendly solution for medical image analysis, complete with a Streamlit-based web interface for seamless interaction.

🌟 Project Overview
This project leverages deep learning to provide reliable brain tumor classification, enabling medical professionals and researchers to analyze MRI scans efficiently. Key highlights include:

High Accuracy: Achieves 95.8% accuracy in tumor detection and classification.
User-Friendly Interface: Built with Streamlit for easy image uploads and result visualization.
Comprehensive Analysis: Displays prediction probabilities and detailed reports.
Real-Time Processing: Processes MRI scans in real time for immediate results.
Open Source: Licensed under the MIT License for community collaboration.


🚀 Features

Advanced CNN Architecture: Built on VGG16 with custom layers for enhanced performance.
Real-Time Classification: Processes MRI scans instantly for quick diagnosis.
Probability Visualization: Displays confidence scores for each tumor category.
Web Interface: Intuitive Streamlit app for uploading and analyzing images.
Detailed Reports: Generates comprehensive analysis for each prediction.
Scalable Design: Easily adaptable for other medical imaging tasks.


🛠 Technical Architecture
Model Architecture
The model is based on the VGG16 architecture (pretrained on ImageNet) with custom modifications for brain tumor classification:

Base Model: VGG16 (pretrained weights)
Custom Layers:
Global Average Pooling 2D
Dense Layer (1024 units, ReLU activation)
Dropout Layer (0.5)
Dense Layer (512 units, ReLU activation)
Output Layer (4 units, Softmax activation)



Dependencies
The project relies on the following Python packages:
numpy
pandas
tensorflow
keras
streamlit
opencv-python
pillow
scikit-learn
matplotlib
seaborn


📦 Installation and Setup
Prerequisites

Python: Version 3.8 or higher
Hardware:
CUDA-compatible GPU (recommended for training)
Minimum 8GB RAM
50GB free disk space



Installation Steps

Clone the Repository:
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection


Create and Activate a Virtual Environment:
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


Install Dependencies:
pip install -r requirements.txt




🖥 Usage
Running the Web Application

Start the Streamlit server:
streamlit run app/main.py


Open your browser and navigate to http://localhost:8501.

Upload an MRI scan image through the web interface.

View the classification results and probability distribution.


Using the Model via Python API
from app.preprocessing import preprocess_image
from models.model import BrainTumorModel

# Initialize the model
model = BrainTumorModel()
model.load_weights('models/weights/brain_tumor_model.keras')

# Preprocess an image
image = preprocess_image('path/to/image.jpg')

# Get prediction
prediction = model.predict(image)


🧠 Model Training
Dataset

Total Images: 3,000
Categories: Glioma, Meningioma, Pituitary Tumor, No Tumor
Image Resolution: 224x224 pixels
Color Space: RGB
Data Split: 70% Training, 15% Validation, 15% Testing

Training Process

Data Preprocessing:

Resize images to 224x224 pixels
Normalize pixel values to [0, 1]
Apply data augmentation (rotation, flip, zoom)


Training Configuration:

Optimizer: Adam (learning rate: 0.0001)
Loss Function: Categorical Cross-entropy
Batch Size: 32
Epochs: 100
Early Stopping: Patience of 10 epochs


Performance Metrics:

Accuracy: 95.8%
Precision: 94.7%
Recall: 95.2%
F1-Score: 94.9%




🤝 Contributing
We welcome contributions to enhance this project! To contribute:

Fork the repository.
Create a feature branch:git checkout -b feature/AmazingFeature


Commit your changes:git commit -m 'Add some AmazingFeature'


Push to the branch:git push origin feature/AmazingFeature


Open a Pull Request.


📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

📚 Citation
If you use this project in your research, please cite:
@software{brain_tumor_detection,
  author = {Your Name},
  title = {Brain Tumor Detection Using Deep Learning},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/brain-tumor-detection}
}


📬 Contact
For questions or feedback, reach out to the project maintainer:Email: azizbahloul3@gmail.com

🙏 Acknowledgments

Dataset Provider: [Institution Name]
Research Reference: [Paper Reference]
Contributors: Special thanks to all contributors who helped make this project possible.
