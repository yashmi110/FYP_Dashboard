import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


class Defence:

    def apply_defence(self, X_train, x_train_adv, y_train, y_train_adv, X_test):

        new_x_train = np.append(X_train, x_train_adv, axis=0)
        new_y_train = np.append(y_train, y_train_adv, axis=0)

        # Scale the input data using StandardScaler or any other suitable scaling technique
        scaler = StandardScaler()
        X_train = scaler.fit_transform(new_x_train)
        X_test = scaler.transform(X_test)

        # Define your model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(new_x_train.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(new_x_train, new_y_train, epochs=10, batch_size=32)

        X_test_defended = ibp_defense(x_train_adv, X_train)

        # Evaluate the accuracy on the defended test set
        test_loss, test_accuracy = model.evaluate(X_test_defended, new_y_train[793:])
        print(f"Defended Accuracy: {test_accuracy}")


# Define the interval bound propagation (IBP) defense function
def ibp_defense(X,X_train):
    # Define the lower and upper bounds
    lower_bound = X_train.min(axis=0)
    upper_bound = X_train.max(axis=0)

    # Apply the bounds to the input data
    X_defended = tf.clip_by_value(X, lower_bound, upper_bound)

    return X_defended