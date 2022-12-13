import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

def print_metrics(y_test, y_pred, classes):
    print(classification_report(y_test, y_pred))

    #cm = confusion_matrix(y_test, y_pred, labels=classes)
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    #disp.plot(cmap=plt.cm.Blues)
    #plt.show()

