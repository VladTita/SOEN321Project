# Adversarial Evasion Attack on Intrusion Detection System

## Overview
This project evaluates the impact of adversarial examples on a neural network trained for intrusion detection. The model classifies network traffic as **benign** or **malicious**, and its performance is tested against adversarial attacks generated using the Fast Gradient Sign Method (FGSM).

---

## Steps to Run

### 1. Install Dependencies
Install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Preprocess the Data
Run the preprocessing script to clean, encode, and normalize the dataset:
```bash
python src/preprocess.py
```

### 3. Train the Model
Train the neural network on the preprocessed data:
```bash
python src/train_model.py
```

### 4. Generate Adversarial Examples
Create adversarial examples to test the model’s robustness:
```bash
python src/generate_attack.py
```

### 5. Evaluate the Model
Test the model’s performance on the adversarial examples:
```bash
python src/evaluate.py
```

---

## Outputs
1. **Baseline Performance**: Printed after running `train_model.py`.
2. **Adversarial Examples**: Saved as `data/adversarial_examples.npy`.
3. **Adversarial Evaluation**: Printed after running `evaluate.py`.

---

## Files
- **`data/Tuesday-WorkingHours.csv`**: Original dataset.
- **`data/processed_data.csv`**: Preprocessed dataset.
- **`src/`**: Contains scripts for preprocessing, training, attack generation, and evaluation.

