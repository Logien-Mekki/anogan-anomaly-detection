# ANOGAN Anomaly Detection

This repository contains a Python implementation of an anomaly detection system using the ANOGAN (Adversarial Network for Anomaly Detection) architecture. The model is trained on the MNIST dataset to identify anomalies based on the normal class.

## Project Overview

The project implements the following:
- **Generator** and **Discriminator** for ANOGAN architecture.
- Training the model using the MNIST dataset, where anomalies are defined as a specific class (e.g., class 3).
- Evaluation of the model using ROC-AUC scores for anomaly detection.

## Requirements

To run the code, you will need the following libraries:
- `numpy`
- `tensorflow`
- `scikit-learn`
- `pandas`

You can install the required libraries by running:

```bash
pip install -r requirements.txt
