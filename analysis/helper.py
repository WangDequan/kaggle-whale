import cv2
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import os
import copy
import numpy as np

def display_image(file_name,base_path=''):
    full_img_path=os.path.join(base_path,file_name)
    img=cv2.imread(full_img_path)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)

def load_image(file_name,base_path,greyscale=False):
    full_img_path=os.path.join(base_path,file_name)
    img=cv2.imread(full_img_path)
    if greyscale:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def convert_digits_prediction_to_file(classified_items,labels_dict):
    classified_items=copy.deepcopy(classified_items)
    #labels=data = [line.strip() for line in open(labels_file, 'r')]
    labels=labels_dict.keys()

    results=[]
    for (img_name,predictions) in classified_items.items():
        #normalize predictions by 1
        for prediction,score in predictions.items():
            predictions[prediction]=score/100

        #add extra labels
        for label in labels:
            if label not in predictions:
                predictions[label]=0

        img_result=predictions
        img_result['Image']= img_name

        results.append(img_result)

    current_wd = os.getcwd()

    labeled_file = os.path.join(current_wd,'digits_predictions.txt')
    print(labeled_file)

    results_df= pd.DataFrame(results)
    results_df.to_csv(labeled_file,index=False)

    return results_df


def convert_digits_prediction_to_list(classified_items,labels_dict):
    classified_items=copy.deepcopy(classified_items)
    labels=labels_dict.keys()
    results=[]
    for (img_name,predictions) in classified_items.items():
        #normalize predictions by 1
        for prediction,score in predictions.items():
            predictions[prediction]=score/100

        scores=[]
        #add extra labels
        for label in labels:
            if label not in predictions:
                predictions[label]=0
            scores.append(predictions[label])

        img_result={}
        img_result['Image']= img_name
        img_result['scores']= scores

        results.append(img_result)
    return results



def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss
