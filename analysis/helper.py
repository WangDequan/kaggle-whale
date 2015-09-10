import cv2
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import os
import copy

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

def convert_digits_prediction_to_file(classified_items,labels_file):
    classified_items=copy.deepcopy(classified_items)
    labels=data = [line.strip() for line in open(labels_file, 'r')]

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
