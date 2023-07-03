import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.utils import shuffle
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ZooAttack, DeepFool, ElasticNet
from art.attacks.evasion import ZooAttack, DeepFool, ElasticNet, VirtualAdversarialMethod, UniversalPerturbation,HopSkipJump

from io import BytesIO
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import InMemoryUploadedFile
from ..models import *
from matplotlib import pyplot as plt_curve
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Define the data augmentation function
def augment_data(data):
    # Randomly permute the rows of the data
    data_permuted = shuffle(data, random_state=42)

    # Add random noise to the data
    data_noisy = data_permuted + np.random.normal(0, 0.05, size=data_permuted.shape)

    return data_noisy


class Attack:
    def generate_adversarial_samples(self, data, attack_type, attack_model):

        print(data)
        if attack_type == "zoo":
            art_classifier = SklearnClassifier(model=attack_model)
            zoo = ZooAttack(classifier=art_classifier, confidence=0.0, targeted=False, learning_rate=1e-1, max_iter=30,
                            binary_search_steps=20, initial_const=1e-3, abort_early=True, use_resize=False,
                            use_importance=False, nb_parallel=10, batch_size=1, variable_h=0.25)
            attack = zoo

        elif attack_type == "en":
            art_classifier = SklearnClassifier(model=attack_model)
            va = VirtualAdversarialMethod(classifier=art_classifier, max_iter=10,
                                          finite_diff=1e-06, eps=0.1,
                                          batch_size=1)

            attack = va

        elif attack_type == "dp":
            art_classifier = SklearnClassifier(model=attack_model)
            dp = HopSkipJump(classifier=art_classifier, batch_size=64, targeted=False,
                             norm=2, max_iter=50, max_eval=10000,
                             init_eval=100, init_size=100, verbose=True)
            attack = dp

        else:
            print("Attack type not support")
            return

        attack_potion = 5

        if data.shape[0] > attack_potion:
            attack_set = data[:attack_potion]
            data_adv = attack.generate(attack_set)

            target_shape = data.shape

            augmented_data = np.concatenate(
                [augment_data(data_adv) for _ in range(target_shape[0] // data_adv.shape[0] + 1)], axis=0)[
                             :target_shape[0], :]
            return augmented_data
        else:
            data_adv = attack.generate(data)
            return data_adv

    def attack_evaluation(self, x_train_adv, y_train, model):

        y_train_adv = model.predict(x_train_adv)

        acc = accuracy_score(y_train, y_train_adv)

        prec = precision_score(y_train, y_train_adv)

        rec = recall_score(y_train, y_train_adv)

        f1 = f1_score(y_train, y_train_adv)

        fpr, tpr, _ = metrics.roc_curve(y_train, y_train_adv)
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
        conf_matrix = confusion_matrix(y_train, y_train_adv)
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

        return acc, prec, rec, f1,image_instance.image, image_instance.cm
