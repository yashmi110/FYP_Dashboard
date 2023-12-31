from FYP_Dashboard.Utility.MLModels.DTree import DTree
from FYP_Dashboard.Utility.MLModels.KNN import Knn
from FYP_Dashboard.Utility.MLModels.RandomForest import RandomForest


class ModelTraining:

    selectedTrainingModel = ""
    trainModel = ""

    X_train, X_test, yTrain, yTest = [], [], [], []

    def training_model(self):

        if self.selectedTrainingModel == "knn":
            print("Knn call")
            knn_obj = Knn(self.X_train, self.yTrain, self.yTest, self.X_test)
            knn, acc, prec, rec, f1, n_roc, n_cm = knn_obj.model_train()
            self.trainModel = knn

            return knn, acc, prec, rec, f1, n_roc, n_cm

        elif self.selectedTrainingModel == "rf":
            print("rfc call")
            rfc_obj = RandomForest(self.X_train, self.yTrain, self.yTest, self.X_test)
            rfc, acc, prec, rec, f1, n_roc, n_cm  = rfc_obj.model_train()
            self.trainModel = rfc

            return rfc, acc, prec, rec, f1, n_roc, n_cm

        elif self.selectedTrainingModel == "dt":
            print("dt call")
            dt_obj = DTree(self.X_train, self.yTrain, self.yTest, self.X_test)
            dt, acc, prec, rec, f1, n_roc, n_cm  = dt_obj.model_train()
            self.trainModel = dt

            return dt, acc, prec, rec, f1, n_roc, n_cm