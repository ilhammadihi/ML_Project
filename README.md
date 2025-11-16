# 🫀 Heart Disease Prediction - End-to-End ML Project

## 📋 Project Overview

This project implements a complete machine learning pipeline to predict the presence of heart disease in patients based on clinical parameters. The project covers all stages of the ML lifecycle: data exploration, preprocessing, model training, evaluation, and deployment.

Author:ILHAM MADIHI   
  
Element: Machine Learning INE2-DATA 2025

---

## 🎯 Objective

Develop a machine learning model to predict heart disease (binary classification) with high accuracy and interpretability for medical decision support.

---

## 📊 Dataset

- **Source**: UCI Heart Disease Dataset / Kaggle
- **Size**: 303 patients
- **Features**: 13 clinical parameters
- **Target**: Binary (0 = No Disease, 1 = Disease)

### Features Description
- age: Age in years
- sex: Sex (1 = male, 0 = female)
- cp: Chest pain type (0-3)
- trestbps: Resting blood pressure
- chol: Serum cholesterol
- fbs: Fasting blood sugar
- restecg: Resting electrocardiographic results
- thalach: Maximum heart rate achieved
- exang: Exercise induced angina
- oldpeak: ST depression induced by exercise
- slope: Slope of the peak exercise ST segment
- ca: Number of major vessels colored by fluoroscopy
- thal: Thalassemia type

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda

### 1. Clone the Repository

git clone https://github.com/ilhammadihi/ML_Project.git
cd ML_Project

## Install Dependencies

pip install -r requirements.txt

##Deployment (Streamlit App)

streamlit run app/streamlit_app.py
