"""
This file includes helper functions (namely get_similarity_matrix) that are used in the threshold selection section in our paper. 
Given a series of clean images & harmful queries, the threshold τ is selected by letting majority of them passes with hyperparameter r (As is implemented in defender.plot_tpr_fpr).
"""
from transformers import AutoModelForPreTraining, AutoTokenizer, AutoImageProcessor
import torch
from PIL import Image
import os, csv, yaml
import numpy as np
import pandas as pd
import tqdm

from defender import plot_tpr_fpr
from utils import generate_denoised_img, get_similarity_list

with open("./settings/settings.yaml") as f:
    settings = yaml.safe_load(f)

def compute_cosine(a_vec , b_vec):
    """calculate cosine similarity"""
    a_vec = a_vec.to("cpu").detach().numpy()
    b_vec = b_vec.to("cpu").detach().numpy()
    norms1 = np.linalg.norm(a_vec, axis=1)
    norms2 = np.linalg.norm(b_vec, axis=1)
    dot_products = np.sum(a_vec * b_vec, axis=1)
    cos_similarities = dot_products / (norms1 * norms2) # ndarray with size=1
    return cos_similarities[0]

def get_similarity_matrix(
    DEVICE = "cuda:0",
    image_dir = "./data/img/testset_denoised",
    text_file = "./data/text/testset.csv",
    cossim_file = "./data/cossim/similarity_matrix.csv"
    ):
    """
    calculate the cosine similarity matrix between each text and denoised images and save the result as csv.

    input:
        :text_file: csv file without headers.
    return: 
    np.ndarray
        :width: number of images in the dir
        :height: len(text_embed_list)
    """
    cossim = get_similarity_list(text_file,"combine",image_dir,DEVICE)
    # flatten dim1 and dim2, make it to 2d-array
    matrix = np.reshape(cossim, (cossim.shape[0], -1))

    # add column name
    img_names = [os.path.splitext(img)[0] for img in sorted(os.listdir(image_dir))]
    tot = np.concatenate((np.array(img_names).reshape(1,-1), matrix), axis=0)
    # save the full similarity matrix as csv
    t = pd.DataFrame(tot)
    t_dir = os.path.dirname(cossim_file)
    if not os.path.exists(t_dir):
        os.makedirs(t_dir)
    t.to_csv(cossim_file, header=False, index=False)
    print(f"csv file saved at: {cossim_file}")

    # analysis
    avg1 = np.mean(matrix.flatten())
    std1 = np.std(matrix.flatten())
    print("cos-sim values avg:{}\tstd:{}".format(avg1,std1))


if __name__=="__main__":
    deviceid = 3
    dataset = "valset"

    # generate clean image cossim file for threshold selection
    generate_denoised_img( model="diffusion",
        imgpth=[os.path.join("./data/img/clean",x) for x in os.listdir("./data/img/clean")],
        save_dir="./temp/img/clean_denoised",
        cps=8,device=deviceid)
    get_similarity_matrix(
        image_dir="./temp/img/clean_denoised",
        text_file="./data/text/valset.csv",
        cossim_file=f"./output/{dataset}_analysis/simmatrix_clean_val.csv",
        DEVICE=f"cuda:{deviceid}")

    fin1 = f"./data/img/{dataset}"
    fout1 = f"./temp/img/{dataset}_denoised"
    fin2 = fout1
    textf = f"./data/text/valset.csv"
    fout2 = f"./output/{dataset}_analysis/simmatrix_{dataset}.csv"
    # generate valset/testset cossim file to evaluate performance
    generate_denoised_img( model="diffusion",
        imgpth=[os.path.join(fin1,x) for x in os.listdir(fin1)],
        save_dir=fout1,cps=8,device=deviceid)
    get_similarity_matrix(
        image_dir=fin2,text_file=textf,
        cossim_file=fout2,DEVICE=f"cuda:{deviceid}")

    # once we have clean image data and {dataset} for evaluation, we could plot the tpr-fpr plot
    plot_tpr_fpr(
        datapath=fout2,
        savepath=f"./output/{dataset}_analysis/tpr-fpr.jpg",trainpath=f"./output/{dataset}_analysis/simmatrix_clean_val.csv",percentage=95)