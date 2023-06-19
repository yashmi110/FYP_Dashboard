import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import tensorflow as tf

class Defense:

    def __init__(self):
        pass

    def adversarial_training_defense(self, selected_train_model, x_train, x_train_adv, y_train, y_train_adv, x_test, y_test):

        new_x_train = np.append(x_train, x_train_adv, axis=0)
        new_y_train = np.append(y_train, y_train_adv, axis=0)

        self_model = selected_train_model.fit(new_x_train, new_y_train)
        new_yPred = self_model.predict(x_test)

        acc = accuracy_score(y_test, new_yPred)
        prec = precision_score(y_test, new_yPred)
        rec = recall_score(y_test, new_yPred)
        f1 = f1_score(y_test, new_yPred)

        return acc, prec, rec, f1

    def provable_defense(self, selected_train_model, x_train, x_train_adv, y_train, y_train_adv, x_test, y_test):

        new_x_train = np.append(x_train, x_train_adv, axis=0)
        new_y_train = np.append(y_train, y_train_adv, axis=0)

        self_model = selected_train_model.fit(new_x_train, new_y_train)

        lower_bound = x_train.min(axis=0)
        upper_bound = x_train.max(axis=0)

        # Apply the bounds to the input data
        x_test_defended = tf.clip_by_value(x_test, lower_bound, upper_bound)
        x_predict_defended = self_model.predict(x_test_defended)

        acc = accuracy_score(y_test, x_predict_defended)
        prec = precision_score(y_test, x_predict_defended)
        rec = recall_score(y_test, x_predict_defended)
        f1 = f1_score(y_test, x_predict_defended)

        return acc, prec, rec, f1

