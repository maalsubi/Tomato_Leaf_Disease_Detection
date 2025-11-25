# AGRI 108
A powerful deep-learning pipeline for automated tomato leaf disease classification, built using a custom CNN architecture enhanced with Efficient Channel Attention (ECA). This model is designed for high accuracy, fast training, and real-world agricultural deployment.

## 🚀 Key Features

🧠 Custom CNN + ECA Attention
Lightweight model with improved feature extraction for higher accuracy.

🌿 10-Class Tomato Disease Dataset
Over 10,000+ labeled leaf images.

🔧 Full Training Pipeline Included
Data augmentation, callbacks, learning rate scheduling, early stopping, etc.

📊 Strong Evaluation Suite
- Accuracy & loss curves
- Confusion matrix
- Classification report
- Model architecture summary
  
📦 Deployment Ready
Exportable to .h5 and .tflite for mobile or edge deployment.

## 📦 Dataset Description

The primary image dataset used for training is collected from:

🔗 PlantVillage – Tomato Leaf Dataset
https://www.kaggle.com/datasets/charuchaudhry/plantvillage-tomato-leaf-dataset

This dataset contains 10 tomato leaf disease classes, with over 10,000 high-quality labeled images, making it one of the best datasets for plant pathology tasks.

📄 Custom Remedy Dataset

Along with the images, this project includes a manually curated CSV file containing remedies for all tomato diseases in the dataset.

✔ Remedies scraped from official Tamil Nadu government agriculture websites

✔ Includes Integrated Disease Management (IDM) practices

✔ Written in English and dynamically translated into Tamil inside the Streamlit app

✔ Fully aligned with the class names predicted by the model

This custom CSV enhances the project by making it not just a classifier, but a full advisory system for farmers.

## 🧠 Model Architecture
This project uses a custom Convolutional Neural Network (CNN) enhanced with Efficient Channel Attention (ECA) to effectively capture key features from tomato leaf images. The architecture is optimized for high accuracy, fast training, and real-world deployment.

For a detailed view of the full model architecture, [Model Architecture](results/model_summary.png)

## 📄 Documentation & Usage

### 🛠️ Setup Instructions

Clone the repository:

```bash
git clone https://github.com/maalsubi/Tomato_Leaf_Disease_Detection.git
cd Tomato_Leaf_Disease_Detection
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Make sure your dataset folders follow this structure:
```bash
tomato/
 ├── train/
 ├── val/
```
Running the Streamlit App
```bash
streamlit run app.py
```
Finally, upload a tomato leaf image then the system will predict the disease, map it to the correct disease name, display the English remedy and translate the remedy to Tamil.
