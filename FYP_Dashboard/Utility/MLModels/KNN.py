from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


class Knn:

    def __init__(self, xTrain, yTrain, yTest, xTest):
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.yTest = yTest
        self.xTest = xTest

    def model_train(self):

        print("call model_train")

        # random forest model creation
        knn = KNeighborsClassifier(n_neighbors=100)
        knn.fit(self.xTrain, self.yTrain)

        #prediction
        print("start predication")
        yPred = knn.predict(self.xTest)

        print("The model used is KNN")

        acc = accuracy_score(self.yTest, yPred)
        print("The accuracy is {}".format(acc))

        prec = precision_score(self.yTest, yPred)
        print("The precision is {}".format(prec))

        rec = recall_score(self.yTest, yPred)
        print("The recall is {}".format(rec))

        f1 = f1_score(self.yTest, yPred)
        print("The F1-Score is {}".format(f1))

        MCC = matthews_corrcoef(self.yTest, yPred)
        print("The Matthews correlation coefficient is {}".format(MCC))

        return knn, acc, prec, rec, f1