import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted label',
           ylabel='True label')
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def show_confusion_matrix(y_true, y_pred, class_names=None):
    if class_names is None:
        class_names = ['Benign', 'Attack']
    plot_confusion_matrix(y_true, y_pred, classes=class_names, normalize=True, cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.show()
    print('\n=== Classification report ===')
    print(classification_report(y_true, y_pred, target_names=class_names))
    print('\n=== Raw confusion matrix ===')
    print(confusion_matrix(y_true, y_pred))
    print()
    