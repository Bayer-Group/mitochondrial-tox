import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_predictions(ytrue, yprob):
    ''' 
    Evaluate the predictions from the NN and RF models
    :param ytrue: true class labels (y)
    :param yprob: predicted proabilities for y
    :return: dataframe with all evaluated metrics
    '''
    auc, acc, mcc, prec, rec, f1, spec = scores(ytrue, yprob)
    results = pd.DataFrame({'AUC': auc, 'MCC': mcc, 'F1 score': f1, 'Precision': prec, 'Recall': rec, 'Specificity':spec}, index=[0])
    return results

def scores(ytrue, yprob):
    '''
    Binary classification model evaluation
    :param ytrue: true class labels (y)
    :param yprob: predicted proabilities for y
    :return: all evaluated metrics
    '''
    ypred = (yprob >= 0.5).astype(int)

    try:
        auc = roc_auc_score(ytrue, yprob)
    except ValueError:
        auc = np.nan
    mcc = matthews_corrcoef(ytrue, ypred)
    acc = accuracy_score(ytrue, ypred)
    prec = precision_score(ytrue, ypred)
    rec = recall_score(ytrue, ypred)
    f1 = f1_score(ytrue, ypred)
    spec = recall_score(ytrue, ypred, pos_label=0)
    
    return auc, acc, mcc, prec, rec, f1, spec