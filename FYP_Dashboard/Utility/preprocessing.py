import pandas as pd


class Preprocessing:
    df = pd.DataFrame()
    dataPercentage = 0.1

    selectedParameter = 'Class'
    selectedTrainingModel = "rf"
    selectedAttackType = "zoo"
    selectedDefenceType = "adv_train"

    trainModel = ""
    X_train_var, X_test_var, yTrain, yTest, x_train_adv, x_test_adv = [], [], [], [], [], []