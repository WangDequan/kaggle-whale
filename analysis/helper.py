import cv2
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import os

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

    if greyscale:
        img=cv2.
