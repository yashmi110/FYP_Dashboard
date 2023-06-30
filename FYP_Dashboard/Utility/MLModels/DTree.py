from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn import tree

from io import BytesIO
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import InMemoryUploadedFile
from matplotlib import pyplot as plt_curve
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns

from ...models import ImageModel


class DTree:

    def __init__(self, xTrain, yTrain, yTest, xTest):
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.yTest = yTest
        self.xTest = xTest

    def model_train(self):

        print("call model_train")

        # random forest model creation
        dTree = tree.DecisionTreeClassifier()
        dTree.fit(self.xTrain, self.yTrain)

        #prediction
        print("start predication")
        yPred = dTree.predict(self.xTest)

        print("The model used is DTree")

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

        fpr, tpr, _ = metrics.roc_curve(self.yTest, yPred)
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
        conf_matrix = confusion_matrix(self.yTest, yPred)
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

        return dTree, acc, prec, rec, f1, image_instance.image, image_instance.cm