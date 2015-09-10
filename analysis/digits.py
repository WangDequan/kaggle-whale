import pandas as pd
import numpy as np
import operator
import os
import requests
import StringIO
import json

class DigitsServerClassification(object):
    def __init__(self,server_url='http://localhost:5000',job_id=None):
        self.server_url = server_url
        self.job_id = job_id


    def classify_images(self,list_of_image_paths):
        #filename='/home/jshoun01/Data/metis/kaggle_whale/data/imgs_subset/w_164.jpg'


        results={}
        output = StringIO.StringIO()

        chunks=[list_of_image_paths[x:x+5] for x in xrange(0, len(list_of_image_paths), 5)]
        #get all images
        num_chunks=len(chunks)
        for chunk_id,chunk in enumerate(chunks):
            for item in chunk:
                output.write(item+'\n')
            output.seek(0)

            files = {'image_list': output}

            api_url=self.server_url+'/models/images/classification/classify_many.json'
            payload = {'job_id':self.job_id}
            r=requests.post(api_url,data=payload,files=files)

            classified = json.loads(r.content)['classifications']


            for (key,value) in classified.items():#every image
                labels=value
                file_name= key.split('/')[-1]

                img_result={}

                for item in labels:
                    key,value= item[0],float(item[1])
                    img_result[key]=value
                results[file_name]=img_result #save top classification

            print ("Chunk %s of %s"%(chunk_id+1,num_chunks))


        return results

    def get_top_label(self,list_of_image_paths):
        all_predictions=self.classify_images(list_of_image_paths)
        p_map=[]

        for image,predictions in all_predictions.items():
            top_prediction=max(predictions.items(), key=operator.itemgetter(1))[0]

            p_map.append({'image':image, 'predicted':top_prediction})
        return p_map


class DigitsFileCreator(object):
    def __init__(self,label_file,base_path):
        self.label_file=label_file
        self.base_path=base_path

    def finalize(self):
        train_df = pd.read_csv(self.label_file)
        label_name_column=train_df.columns[1]
        path_column=train_df.columns[0]

        train_df['full_path']= self.base_path+"/"+train_df[path_column]

        labels_unique=train_df[label_name_column].unique()
        labels_dict= {key: idx for idx,key in enumerate(labels_unique)}

        current_wd = os.getcwd()

        labeled_file = os.path.join(current_wd,'digits_labels.txt')

        with open(labeled_file, 'w') as outfile:
            for k, v in sorted(labels_dict.items(),key=operator.itemgetter(1)):
                outfile.write(k+"\n")

        train_df['label_id']=train_df[label_name_column].apply(lambda x : labels_dict[x])

        shuffled_df=train_df.iloc[np.random.permutation(np.arange(len(train_df)))]

        ratio=0.75
        train_last=int(len(train_df)*ratio)

        train_split=shuffled_df[0:train_last]
        validation_split=shuffled_df[train_last:]

        train_file = os.path.join(current_wd,'digits_train.csv')

        train_split[['full_path','label_id']].to_csv(train_file,header=False, index=False, sep=" ")

        validation_file = os.path.join(current_wd,'digits_validation.csv')

        validation_split[['full_path','label_id']].to_csv(validation_file,header=False, index=False, sep=" ")

        print labeled_file
        print train_file
        print validation_file
        #return train_split,validation_split
