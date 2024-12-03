import tensorflow as tf
import numpy as np

def create_adversarial_examples(model, X, y, epsilon=0.01):
    # Convert inputs to tensors
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        predictions = model(X_tensor)
        loss = tf.keras.losses.categorical_crossentropy(y_tensor, predictions)

    # Compute gradients and generate perturbations
    gradient = tape.gradient(loss, X_tensor)
    perturbations = epsilon * tf.sign(gradient)
    X_adv = X_tensor + perturbations
    return X_adv.numpy()

if __name__ == "__main__":
    from train_model import train_model
    model, X_test, y_test = train_model("data/processed_data.csv")

    # Generate adversarial examples
    X_adv = create_adversarial_examples(model, X_test, y_test)
    np.save("data/adversarial_examples.npy", X_adv)
    print("Adversarial examples generated and saved.")
