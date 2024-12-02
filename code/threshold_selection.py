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
    cossim_file = "./data/cossim/similarity_matrix.csv",
    injection=False
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
    ########################
    model = AutoModelForPreTraining.from_pretrained(
        settings["Target_model_path"], torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to(DEVICE)# this runs out of memory

    tokenizer = AutoTokenizer.from_pretrained(settings["Target_model_path"])
    imgprocessor = AutoImageProcessor.from_pretrained(settings["Target_model_path"])


    def get_img_embedding(image_path):
        image = Image.open(image_path)
        # img embedding
        pixel_value = imgprocessor(image, return_tensors="pt").pixel_values.to(DEVICE)
        image_outputs = model.vision_tower(pixel_value, output_hidden_states=True)
        selected_image_feature = image_outputs.hidden_states[model.config.vision_feature_layer]
        selected_image_feature = selected_image_feature[:, 1:] # by default
        image_features = model.multi_modal_projector(selected_image_feature)
        # calculate average to compress the 2th dimension
        image_features = torch.mean(image_features, dim=1).detach().to("cpu")
        del pixel_value, image_outputs, selected_image_feature
        torch.cuda.empty_cache()
        return image_features

    def get_text_embedding(text: str):
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
        input_embeds = model.get_input_embeddings()(input_ids)
        # calculate average to get shape[1, 4096]
        input_embeds = torch.mean(input_embeds, dim=1).detach().to("cpu")
        del input_ids
        torch.cuda.empty_cache()
        return input_embeds


    # generate embeddings for images
    img_embed_list = []
    img_names = []
    dir1 = os.listdir(image_dir)
    dir1.sort()
    for img in tqdm.tqdm(dir1,desc="generating img embeddings"):
        ret = get_img_embedding(f"{image_dir}/{img}")
        img_embed_list.append(ret)
        img_names.append(os.path.splitext(img)[0])
    
    # generate embeddings for text
    malicious_text_embed_list = []
    with open(text_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            # only keep FunctionalCategory=standard rows
            if row[1]!="standard":
                continue
            text = row[0]
            ret = get_text_embedding(text)
            malicious_text_embed_list.append(ret)

    # compute cosine similarity between malicious text and images
    malicious_result = np.zeros((len(malicious_text_embed_list), len(img_embed_list)))
    for i in tqdm.trange(len(malicious_text_embed_list),desc="outer loop"):
        text_embed = malicious_text_embed_list[i]
        for j in range(len(img_embed_list)):
            if injection and i!=(j//8)%len(malicious_text_embed_list):
                malicious_result[i, j] = 0 
                continue
            img_embed = img_embed_list[j]
            cos_sim = compute_cosine(img_embed, text_embed)
            malicious_result[i, j] = cos_sim

    # add column name
    tot = np.concatenate((np.array(img_names).reshape(1,-1), malicious_result), axis=0)
    # save the full similarity matrix as csv
    t = pd.DataFrame(tot)
    t_dir = os.path.dirname(cossim_file)
    if not os.path.exists(t_dir):
        os.makedirs(t_dir)
    t.to_csv(cossim_file, header=False, index=False)
    print(f"csv file saved at: {cossim_file}")

    # analysis
    avg1 = np.mean(malicious_result.flatten())
    std1 = np.std(malicious_result.flatten())

    print("cos-sim values avg:{}\tstd:{}".format(avg1,std1))

from defender import plot_tpr_fpr
from utils import generate_denoised_img

if __name__=="__main__":
    deviceid = 2
    dataset = "imageJP"

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
        savepath=f"./output/{dataset}_analysis/tpr-fpr.jpg",trainpath=f"./output/{dataset}_analysis/simmatrix_clean_val.csv")