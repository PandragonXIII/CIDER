"""
a defender class that classify adversarial and clean image with 
a decline threshold
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os

logger = logging.getLogger(__name__)

class Defender():
    """
    given text and image, if any decreace of cosine similarity greater than threshold 
    (the delta values are smaller than threshold) occurs during denoise, 
    then consider it as adversarial
    contains methods:
        1. train: calculate threshold with given clean image data and specified ratio.
        2. predict: given img-text pair, return whether it is adversarial.
        3. get_confusion_matrix: given validation dataset, call predict function and return statistics (i.e. confusion matrix)
    """
    def __init__(self, threshold = None):
        self.threshold = threshold
        logger.info(f"Defender initialized with threshold={self.threshold}")
    
    def train(self, data, ratio=0.9):
        """
        calculate threshold with given data and specified conservative ratio
        :data: cosine similarity between malicious query(text) & clean image 
            cols: denoise times
            rows: different text
        :ratio: the ratio of clean image cosine similarity decrease value that should be lower than threshold
        """
        if type(data) == list:
            for df in data:
                names = [i for i in range(df.shape[1])]
                df.columns = names
            data = pd.concat(data, axis=0, ignore_index=True)
        # get the first col (origin image) as the base cosine similarity
        base = data.iloc[:,0]
        # get the decrease value
        decrease = data.iloc[:,1:].sub(base, axis=0)
        # get the ratio of decrease value that is lower than threshold
        self.threshold = np.percentile(decrease, (1-ratio)*100)
        print(f"Threshold updated to {self.threshold}")

    def predict(self, cossims):
        """
        given a series of cosine similarity , return whether it is adversarial
        by calculating the interpolation between each and the first one

        :cossims: a nD array of cosine similarity (m*n)
            cols: denoise times n
            row: m text
        return: a nD array of boolean (m*1)
            True means adversarial
        """
        if len(cossims.shape) == 1:
            return True in (np.array(cossims[1:]) - cossims[0] < self.threshold)
        else:
            ret = []
            for r in range(cossims.shape[0]):
                row = cossims.iloc[r]
                ret.append(True in (np.array(row.iloc[1:]) - row.iloc[0] < self.threshold))
            return ret
        
    def get_lowest_idx(self, cossims):
        """
        given a series of cosine similarity , return whether it is adversarial
        for adversarial data, return the index of lowest cosine similarity iff 
        the decline value is more than threshold τ, otherwise return 0

        :cossims: a nD array of cosine similarity (1*n)
            cols: denoise times n
            row: 1 text
        return: a nD array of int (1*1)
            0 is clean, positive int is the index of lowest cosine similarity
        """
        # reshape the data
        cossims = np.array(cossims).reshape(1,-1)
        delta = cossims - cossims[0,0]
        if np.min(delta)<self.threshold:
            return np.argmin(delta)
        else:
            return 0
    
    def get_confusion_matrix(self, datapath:str,checkpt_num=8,group=True):
        """test the defender with given cosine similarity data, 
        save the statics to savepath\n
        only consider malicious text\n
        the result contains 4 rows: for each adversarial image, 
        consider it as positive and clean as negative, 
        output the results
        checkpt_num: the maximum number of denoise times checkpoint to consider
        group: if true, output separate matrix for different adv image"""
        df = pd.read_csv(datapath)
        try:
            df = df[df["is_malicious"]==1] # only consider malicious text input
        except:
            pass
        results = {
            "constraint":[],
            "accuracy":[],
            "recall(dsr)":[],
            "precision":[],
            "f1":[],
            "classification threshold":[],
            "fpr":[]
        }
        fp,tn=0,0
        # get the Test Set clean image data for prediction
        all_clean_header = [col for col in df.columns if "clean_" in col]
        clean_classes_names = set(["_".join(h.split("_")[:2]) for h in all_clean_header])
        for clean_class in clean_classes_names:
            clean_header = [col for col in df.columns if clean_class+"_" in col]
            if len(clean_header)>checkpt_num:
                clean_header = clean_header[:checkpt_num]
            clean_data = df[clean_header]
            # predict with clean image
            clean_predict = np.array(self.predict(clean_data))
            fp += sum(clean_predict[:])
            tn += sum(~clean_predict[:])

        tot_tp,tot_fn=0,0
        # get the adversarial image data
        all_adv_header = [col for col in df.columns if "prompt_" in col]
        adv_classes_names = set(["_".join(h.split("_")[:3]) for h in all_adv_header])
        for adv_class in adv_classes_names:
            # list the headers of adv_class constraint
            adv_header = [col for col in df.columns if adv_class+"_" in col]
            if len(adv_header)>checkpt_num:
                adv_header = adv_header[:checkpt_num]
            # get data
            adv_data = df[adv_header]
            # predict
            adv_predict = np.array(self.predict(adv_data))
            tp = sum(adv_predict[:])
            fn = sum(~adv_predict[:])
            tot_tp+=tp
            tot_fn+=fn

            if not group: # calculate together later
                continue
            # if group, calculate matrix for the class now
            # check num positive = negative
            assert tp+fn == fp+tn

            acc = (tp+tn)/(tp+fn+fp+tn)
            recall = tp/(tp+fn)
            precision = tp/(tp+fp)
            f1 = 2*precision*recall/(precision+recall)
            fpr = fp/(fp+tn)
            results["constraint"].append(adv_class.split("_")[-1])
            results["accuracy"].append(acc)
            results["recall(dsr)"].append(recall)
            results["precision"].append(precision)
            results["f1"].append(f1)
            results["classification threshold"].append(self.threshold)
            results["fpr"].append(fpr)  
        if not group:
            # assert tot_tp+tot_fn==fp+tn
            tp = tot_tp
            fn = tot_fn
            acc = (tp+tn)/(tp+fn+fp+tn)
            recall = tp/(tp+fn)
            precision = tp/(tp+fp)
            f1 = 2*precision*recall/(precision+recall)
            fpr = fp/(fp+tn)
            results["constraint"].append("all")
            results["accuracy"].append(acc)
            results["recall(dsr)"].append(recall)
            results["precision"].append(precision)
            results["f1"].append(f1)
            results["classification threshold"].append(self.threshold)
            results["fpr"].append(fpr)  
        results = pd.DataFrame(results)
        return results

def plot_tpr_fpr(datapath:str, savepath:str, cpnum=8, percentage:int|None=None, trainpath = "./src/intermediate-data/10clean_similarity_matrix_val.csv"):
    """
    plot tpr-fpr scatter plot with specific detection & denoise method for evaluation.
    
    inputs: 

    :datapath: csv file containing cosine similarity values of origin & denoised 
    images to malicious queries.
    :savepath: path to save the figure
    :percentage: the chosen point with special annotation
    """
    # training data
    data = pd.read_csv(trainpath)
    train_data = []
    print(f"num of clean images: {data.shape[1]//cpnum}")
    for i in range(data.shape[1]//cpnum): # the number of clean images
        train_data.append(data.iloc[:,i*cpnum:(i+1)*cpnum])
    detector = Defender()

    datapoints = []
    for i in range(80,101):
        detector.train(train_data,ratio=i/100)
        # predict on test data and return results
        confusion = detector.get_confusion_matrix(datapath,checkpt_num=cpnum,group=False)
        # delete the row of inf
        confusion = confusion[confusion["constraint"]!="inf"]
        point = confusion[["recall(dsr)","fpr"]].mean(axis=0)
        if i==percentage:
            chosen_pt=point
        else:
            datapoints.append(point)
    datapoints = pd.concat(datapoints, axis=1, ignore_index=True)

    # plot
    plt.clf()
    plt.scatter(datapoints.loc["fpr"],datapoints.loc["recall(dsr)"],s=80,c="#23bac5",alpha=0.7,zorder=2)
    # annotate a specific point as a cross.
    if percentage is not None:
        plt.scatter(chosen_pt[1],chosen_pt[0],s=150,c="#fd763f",marker="X",
                    linewidth=1,edgecolors='w',zorder=3)
        plt.annotate(f"{percentage}%",(chosen_pt[1],chosen_pt[0]),xytext=(chosen_pt[1]-0.02,chosen_pt[0]+0.05),xycoords="data")

    plt.xlabel("False Positive Rate",fontsize=11)
    plt.ylabel("True Positive Rate",fontsize=11)

    plt.yticks(np.arange(0,1,0.2),['{:.0%}'.format(_) for _ in np.arange(0,1,0.2)])
    plt.xticks(np.arange(0,1,0.2),['{:.0%}'.format(_) for _ in np.arange(0,1,0.2)])

     
    plt.tight_layout()
    plt.savefig(os.path.splitext(savepath)[0]+".jpeg")
    plt.savefig(os.path.splitext(savepath)[0]+".pdf")
    print(datapoints.loc["fpr"],datapoints.loc["recall(dsr)"])
    print(f"tpr-fpr plot saved at {os.path.splitext(savepath)[0]}")
