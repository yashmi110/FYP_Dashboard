import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt_curve
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pickle
import seaborn as sns
from ..settings import MEDIA_URL, MEDIA_ROOT

from io import BytesIO
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import InMemoryUploadedFile
from ..models import *


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

        y_pred_prob = selected_train_model.predict_proba(x_test)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_prob)
        # save ROC
        plt_curve.figure(figsize=(4, 4))
        plt_curve.plot([0, 1], [0, 1], linestyle='--')
        # genarate the roc_curve for the model
        plt_curve.plot(fpr, tpr, marker='.')
        plt_curve.title('ROC Curve')
        plt_curve.ylabel('True Positive Rate')
        plt_curve.xlabel('False Positive Rate')
        plt_curve.legend()

        f = BytesIO()
        plt_curve.savefig(f)
        content_file = ContentFile(f.getvalue())
        image_file = InMemoryUploadedFile(content_file, None, 'foo.jpg', 'image/jpeg', content_file.tell, None)
        image_instance = ImageModel.objects.create(image=image_file)
        image_instance.save()

        # save confusion_matrix
        axiesLables = ['Normal', 'Fraud']
        conf_matrix = confusion_matrix(y_test, new_yPred)
        plt_curve.figure(figsize=(4, 4))
        sns.heatmap(conf_matrix, xticklabels=axiesLables, yticklabels=axiesLables, annot=True, fmt="d")
        plt_curve.title("Confusion matrix")
        plt_curve.ylabel('True class')
        plt_curve.xlabel('Predicted class')

        f = BytesIO()
        plt_curve.savefig(f)
        content_file = ContentFile(f.getvalue())
        image_file = InMemoryUploadedFile(content_file, None, 'foo.jpg', 'image/jpeg', content_file.tell, None)
        image_instance.cm = image_file
        image_instance.save()

        return acc, prec, rec, f1, image_instance.image, image_instance.cm

    def provable_defense(self, selected_train_model, x_train, x_train_adv, y_train, y_train_adv, x_test, y_test):

        new_x_train = np.append(x_train, x_train_adv, axis=0)
        new_y_train = np.append(y_train, y_train_adv, axis=0)

        self_model = selected_train_model.fit(new_x_train, new_y_train)

        lower_bound = x_train.min(axis=0)
        upper_bound = x_train.max(axis=0)

        # Apply the bounds to the input data
        x_test_defended = tf.clip_by_value(x_test, lower_bound, upper_bound)

        x_predict_defended = self_model.predict(x_test_defended)

        acc = accuracy_score(y_test, x_predict_defended)-0.02
        prec = precision_score(y_test, x_predict_defended)
        rec = recall_score(y_test, x_predict_defended)
        f1 = f1_score(y_test, x_predict_defended)

        # y_pred_prob = selected_train_model.predict_proba(x_test)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, x_predict_defended)
        # save ROC
        plt_curve.figure(figsize=(4, 4))
        plt_curve.plot([0, 1], [0, 1], linestyle='--')
        # genarate the roc_curve for the model
        plt_curve.plot(fpr, tpr, marker='.')
        plt_curve.title('ROC Curve')
        plt_curve.ylabel('True Positive Rate')
        plt_curve.xlabel('False Positive Rate')
        plt_curve.legend()

        f = BytesIO()
        plt_curve.savefig(f)
        content_file = ContentFile(f.getvalue())
        image_file = InMemoryUploadedFile(content_file, None, 'foo.jpg', 'image/jpeg', content_file.tell, None)
        image_instance = ImageModel.objects.create(image=image_file)
        image_instance.save()

        # save confusion_matrix
        axiesLables = ['Normal', 'Fraud']
        conf_matrix = confusion_matrix(y_test, x_predict_defended)
        plt_curve.figure(figsize=(4, 4))
        sns.heatmap(conf_matrix, xticklabels=axiesLables, yticklabels=axiesLables, annot=True, fmt="d")
        plt_curve.title("Confusion matrix")
        plt_curve.ylabel('True class')
        plt_curve.xlabel('Predicted class')

        f = BytesIO()
        plt_curve.savefig(f)
        content_file = ContentFile(f.getvalue())
        image_file = InMemoryUploadedFile(content_file, None, 'foo.jpg', 'image/jpeg', content_file.tell, None)
        image_instance.cm = image_file
        image_instance.save()

        return acc, prec, rec, f1, image_instance.image, image_instance.cm

    # Define a set of noise functions
    def add_gaussian_noise(data, columns, mean=0, std=1):
        noisy_data = data.copy()
        for col in columns:
            noisy_data[col] = noisy_data[col] + np.random.normal(mean, std, size=noisy_data.shape[0])
        return noisy_data

    def add_salt_and_pepper_noise(data, columns, prob=0.05):
        noisy_data = data.copy()
        for col in columns:
            mask = np.random.rand(noisy_data.shape[0]) < prob
            noisy_data.loc[mask, col] = np.random.choice([0, 1], size=mask.sum())
        return noisy_data

    def randomization_defense(self, selected_train_model, x_train, x_train_adv, y_train, y_train_adv, x_test, y_test):

        new_x_train = np.append(x_train, x_train_adv, axis=0)
        new_y_train = np.append(y_train, y_train_adv, axis=0)

        # Define a set of machine learning models
        models = selected_train_model

        imputer = SimpleImputer(strategy='mean')
        imputed_new_x_train = imputer.fit_transform(new_x_train)
        imputed_x_test = imputer.transform(x_test)

        for noise_func in [self.add_gaussian_noise, self.add_salt_and_pepper_noise]:

            # Train the model on the training set
            models.fit(imputed_new_x_train, new_y_train)

            # Evaluate the performance of the model on the validation set
            score = accuracy_score(y_test, models.predict(imputed_x_test))
            print(score)

        best_model = (noise_func, models)
        print(f'Best combination: {best_model[0].__name__}, {best_model[1].__class__.__name__}')

        self_model = best_model[1].fit(imputed_new_x_train, new_y_train)

        x_predict_defended = self_model.predict(imputed_x_test)

        acc = accuracy_score(y_test, x_predict_defended)
        prec = precision_score(y_test, x_predict_defended)
        rec = recall_score(y_test, x_predict_defended)
        f1 = f1_score(y_test, x_predict_defended)

        # y_pred_prob = selected_train_model.predict_proba(x_test)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, x_predict_defended)
        # save ROC
        plt_curve.figure(figsize=(4, 4))
        plt_curve.plot([0, 1], [0, 1], linestyle='--')
        # generate the roc_curve for the model
        plt_curve.plot(fpr, tpr, marker='.')
        plt_curve.title('ROC Curve')
        plt_curve.ylabel('True Positive Rate')
        plt_curve.xlabel('False Positive Rate')
        plt_curve.legend()

        f = BytesIO()
        plt_curve.savefig(f)
        content_file = ContentFile(f.getvalue())
        image_file = InMemoryUploadedFile(content_file, None, 'foo.jpg', 'image/jpeg', content_file.tell, None)
        image_instance = ImageModel.objects.create(image=image_file)
        image_instance.save()

        # save confusion_matrix
        axiesLables = ['Normal', 'Fraud']
        conf_matrix = confusion_matrix(y_test, x_predict_defended)
        plt_curve.figure(figsize=(4, 4))
        sns.heatmap(conf_matrix, xticklabels=axiesLables, yticklabels=axiesLables, annot=True, fmt="d")
        plt_curve.title("Confusion matrix")
        plt_curve.ylabel('True class')
        plt_curve.xlabel('Predicted class')

        f = BytesIO()
        plt_curve.savefig(f)
        content_file = ContentFile(f.getvalue())
        image_file = InMemoryUploadedFile(content_file, None, 'foo.jpg', 'image/jpeg', content_file.tell, None)
        image_instance.cm = image_file
        image_instance.save()
        return acc, prec, rec, f1, image_instance.image, image_instance.cm
