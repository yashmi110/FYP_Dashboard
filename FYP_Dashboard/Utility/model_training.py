from FYP_Dashboard.Utility.MLModels.KNN import Knn


class ModelTraining:

    selectedTrainingModel = "rf"
    trainModel = ""

    X_train, X_test, yTrain, yTest = [], [], [], []

    def training_model(self):

        if self.selectedTrainingModel == "knn":
            print("knn call")
            knn_obj = Knn(self.X_train, self.yTrain, self.yTest, self.X_test)
            knn, acc, prec, rec, f1 = knn_obj.model_train()
            self.trainModel = knn

            return knn, acc, prec, rec, f1