# Brain Stroke Prediction

A machine learning-based web application that predicts whether a person is likely to have a brain stroke or not based on the provided input features.

## Project Overview

This project focuses on developing a web application for brain stroke prediction using Machine Learning. The application analyzes the provided input data and uses trained classification models to predict whether the input indicates a stroke or a normal condition.

Multiple classification algorithms are explored and compared to identify a suitable model for the prediction task.

The application provides a user-friendly web interface where users can enter the required information and receive a prediction result.

## Machine Learning Algorithms

The project explores the following classification algorithms:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- ExtraTree Classifier
- Gaussian Naive Bayes
- Decision Tree
- Random Forest Classifier

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Flask
- HTML
- CSS

## Features

- Data preprocessing
- Feature scaling
- Machine learning model training
- Classification model comparison
- Stroke prediction
- Flask-based web interface
- Prediction results

## Project Structure

```text
brain-stroke-prediction/
│
├── app.py
├── train_model.py
├── brain.csv
├── req.txt
│
├── index.html
├── result.html
│
├── stroke_model.pkl
├── scaler.pkl
├── preprocessor.pkl
├── imputer.pkl
├── encoders.pkl
│
├── model_comparison.csv
└── model_comparison.png
