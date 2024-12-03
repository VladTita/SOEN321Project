import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf

def evaluate_attack(model, X_adv, y_test):
    # Predict on adversarial examples
    y_pred_adv = model.predict(X_adv)

    # Convert one-hot encoded labels back to discrete labels
    y_test_discrete = np.argmax(y_test, axis=1)  # Convert y_test from one-hot to integers
    y_pred_adv_discrete = np.argmax(y_pred_adv, axis=1)  # Convert predictions to integers

    # Generate classification report
    print("Performance on Adversarial Examples:\n")
    print(classification_report(y_test_discrete, y_pred_adv_discrete))

if __name__ == "__main__":
    from train_model import train_model
    import tensorflow as tf

    # Load model and adversarial examples
    model, _, y_test = train_model("data/processed_data.csv")
    X_adv = np.load("data/adversarial_examples.npy")

    # Evaluate model on adversarial examples
    evaluate_attack(model, X_adv, y_test)
