import os

import pandas as pd
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render

from FYP_Dashboard.Utility.attack import Attack
from FYP_Dashboard.Utility.defense import Defense
from FYP_Dashboard.Utility.explainable_ai import ExplainableAI
from FYP_Dashboard.Utility.feature_selection import FeatureSelection
from FYP_Dashboard.Utility.model_training import ModelTraining

data = {"isUploadDone": False}
feature_selection = FeatureSelection()
model_training = ModelTraining()
attack = Attack()
explainable = ExplainableAI()
defence = Defense()

df = pd.DataFrame()

X_train, X_test, yTrain, yTest = [], [], [], []
x_train_adv, x_test_adv, y_train_adv = [], [], []

selected_train_model = {}

before_attack = []
after_attack = []


def index(request):
    print(request.POST)
    global X_train, X_test, yTrain, yTest, x_train_adv, x_test_adv, selected_train_model, before_attack, after_attack, y_train_adv

    if 'btnUpload' in request.POST:
        uploaded_file = request.FILES['document']

        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        url = fs.url(name)

        df = pd.read_csv(os.getcwd() + url)
        feature_selection.df = df.sample(frac=0.2)

        data["columns"] = df.columns
        data["rows"] = df.shape[0]
        data["isUploadDone"] = True

    elif 'btnFeatureSelection' in request.POST:

        label_parameter = request.POST['parameterSelect']
        feature_selection.className = label_parameter
        feature_selection.split_data()
        feature_selection.evaluate_methods()
        X_train, X_test, yTrain, yTest = feature_selection.find_best_method()

    elif 'btnNormalEnv' in request.POST:
        selected_model = request.POST['modelTypeSelect']
        model_training.selectedTrainingModel = selected_model
        model_training.X_train = X_train
        model_training.X_test = X_test
        model_training.yTrain = yTrain
        model_training.yTest = yTest
        selected_train_model, acc, prec, rec, f1, n_roc, n_cm = model_training.training_model()

        data['acc'] = round((acc * 100), 2)
        data['prec'] = round((prec * 100), 2)
        data['rec'] = round((rec * 100), 2)
        data['f1'] = round((f1 * 100), 2)
        data['n_roc'] = n_roc
        data['n_cm'] = n_cm

        before_attack = explainable.get_values(X_train.values, X_train.columns, selected_train_model, X_test.values[0])

    elif 'btnAttack' in request.POST:

        attack_type = request.POST['attackTypeSelect']
        x_train_adv = attack.generate_adversarial_samples(X_train.values, attack_type, selected_train_model)
        x_test_adv = attack.generate_adversarial_samples(X_test.values, attack_type, selected_train_model)
        after_attack = explainable.get_values(x_train_adv, X_train.columns, selected_train_model, x_test_adv[0])

        y_train_adv = selected_train_model.predict(x_train_adv)
        data["description_list"] = explainable.get_description(before_attack, after_attack, X_train.columns)

        acc, prec, rec, f1,a_roc, a_cm = attack.attack_evaluation(x_train_adv, yTrain, selected_train_model)

        data['a_acc'] = round((acc * 100), 2)
        data['a_prec'] = round((prec * 100), 2)
        data['a_rec'] = round((rec * 100), 2)
        data['a_f1'] = round((f1 * 100), 2)
        data['a_roc'] = a_roc
        data['a_cm'] = a_cm

    elif 'btnDefence' in request.POST:

        defense_type = request.POST['defenseTypeSelect']
        if defense_type == "Trainee":
            acc, prec, rec, f1, roc, cm = defence.adversarial_training_defense(selected_train_model, X_train, x_train_adv,
                                                                      yTrain, y_train_adv, X_test, yTest)
        elif defense_type == "Randomization":
            acc, prec, rec, f1,roc, cm = defence.randomization_defense(selected_train_model, X_train, x_train_adv, yTrain,
                                                          y_train_adv, X_test, yTest)
        else:
            acc, prec, rec, f1,roc, cm = defence.provable_defense(selected_train_model, X_train, x_train_adv, yTrain,
                                                          y_train_adv, X_test, yTest)

        data['d_acc'] = round((acc * 100), 2)
        data['d_prec'] = round((prec * 100), 2)
        data['d_rec'] = round((rec * 100), 2)
        data['d_f1'] = round((f1 * 100), 2)
        data['roc'] = roc
        data['cm'] = cm

    return render(request, 'index.html', data)
