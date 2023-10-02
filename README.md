# Diabetes Prediction Flask Web App

This repository contains the code for a web-based diabetes prediction application using a machine learning model. The application is built using Flask and allows users to input various health parameters to predict the likelihood of diabetes.
## Overview

The web app utilizes a machine learning model, specifically an ensemble model combining RandomForestClassifier, LogisticRegression, and Support Vector Machine (SVM), to provide predictions. The model was trained on the diabetes dataset from kaggle.

## Features

- Users can input their health parameters such as Glucose level, Blood Pressure, Skin Thickness, Insulin level, BMI, Diabetes Pedigree Function (DPF), and Age.
- The ensemble machine learning model predicts the likelihood of diabetes based on the input.
- The app displays the prediction result, indicating whether the user is likely to have diabetes or not.
- Users can receive accurate and quick predictions for early diabetes detection.

## Model

The machine learning model used in this app is an ensemble of RandomForestClassifier, LogisticRegression, and Support Vector Machine (SVM). The ensemble approach combines the strengths of these classifiers to improve prediction accuracy.
