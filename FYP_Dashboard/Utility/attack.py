import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.utils import shuffle
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion import ZooAttack, DeepFool, ElasticNet


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
        else:
            print("Attack type not support")
            return

        attack_potion = 30

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

        return acc, prec, rec, f1
