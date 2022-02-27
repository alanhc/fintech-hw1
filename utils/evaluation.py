import numpy as np
def myScore(y_pred, y_test):
    ct = np.unique( y_test, return_counts=True)[1]
    right = (y_pred==y_test)
    y_pred_c = np.logical_not(y_pred).astype(int)

    right_0 = y_pred_c & right
    right_0 = right_0.sum()
    right_1 = y_pred & right
    right_1 = right_1.sum()
    score = (right_0*1+right_1*9)/(ct[0]*1+ct[1]*9)
    return score

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report,confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def result(model, X, y_true, y_pred):
    cm = confusion_matrix(y_true,y_pred)
    report = classification_report(y_true,y_pred, output_dict=True)
    #print(cm,"\n",report)  
    #print("roc", roc_curve(y_true, y_pred, average=None))
    cv_score = cross_val_score(model, X, y_true, cv=5)
    roc = roc_auc_score(y_true, y_pred, average=None)
    #print("cv score",cv_score.mean() )
    fpr,tpr,threshols=roc_curve(y_true,y_pred)
    plt.plot([0,1],[0,1],"k--")
    plt.plot(fpr,tpr,label='classifier ')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    return cm ,report, cv_score.mean(), roc
