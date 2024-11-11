import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joypy.joyplot
import time


def clean_adv_cossim_difference(csv_path:str, figpath="./figures/cossim_difference.png"):
    """
    use kdeplot to display difference between <clean img,query> and <adv img, query>. both images are original (without denoise)
    """

    df = pd.read_csv(csv_path)
    origin_imgs = df[[col for col in df.columns if "denoised_000times" in col]]

    clean_cols = origin_imgs[[col for col in origin_imgs.columns if "clean_" in col]]
    adv_cols = origin_imgs[[col for col in origin_imgs.columns if "prompt_constrained_" in col]]


    # calculate difference
    cleandata = clean_cols.sum(axis=1)
    advdata = adv_cols.sum(axis=1)

    diff = cleandata-advdata
    diff = pd.DataFrame(diff,columns=["Difference of Cosine Similarity"])


    plt.clf()
    f,ax = plt.subplots(figsize=(4.3,2))
    sns.kdeplot(diff,x="Difference of Cosine Similarity",color = "#eeca40",ax=ax,fill=True,alpha=0.5)
    # plt.legend(["clean","adversarial"])

    plt.tight_layout()
    figpath = os.path.splitext(figpath)[0]
    plt.savefig(figpath+".jpg")
    plt.savefig(figpath+".pdf")

def delta_cos_sim_distribution(csv_path:str, it=250, figpath="./figures/delta_cossim_distribution.png"):
    '''
    visualize (with joyplot-mountain) the **difference** of cosine similarity distribution, show the difference of delta value on cossim.
    **for each single image there is a mountain curve**.
    Delta cossim value = decline in cossim from clean to denoised images with it(250 by default) iterations.

    :input: cosine similarity csv file 
    :output: None
    '''
    # read the csv file
    df = pd.read_csv(csv_path)

    # denoised img & queries cosine similarity
    data = df.filter(regex="{:0>3d}times".format(it))
    # rename: remove "_denoised_xxtimes" suffix
    data.columns = ["_".join(col.split("_")[:-2]) for col in data.columns]
    # origin img & queries cosine similarity
    origin_data = df.filter(regex="000times")
    origin_data.columns = ["_".join(col.split("_")[:-2]) for col in origin_data.columns]
    # calcualte delta value
    delta = data-origin_data

    # plot the distribution, x-axis: cossim, y-axis: frequency
    # Draw Plot
    delta["var"] = "Δ cosine similarity" # add a common column to group by(draw in one axis)
    # set alpha=0.6 to show the overlapping
    colors = ["#F0EFED","#F0EFED","#F0EFED","#F0EFED","#1399B2","#BD0026","#FD8D3C","#F1D756","#EF767B"]
    # colors = ["#E4A031","#D68438","#C76B60","#B55384","#7C4D77","#474769","#26445E","#4C7780","#73A5A2","#F6EC21","#B2B6C1","#D6E2E2"]
    fig, axes = joypy.joyplot(delta,
                              alpha=0.6, color=colors, legend=True, loc="upper left", linecolor="#01010100"
                              )
    plt.title(f"Δ cosine similarity after {it} iters denoising")
    # plt.tight_layout()
    plt.savefig(os.path.splitext(figpath)[0]+".jpg")
    plt.savefig(os.path.splitext(figpath)[0]+".pdf")
    return

def grouped_cossim_distribution(csv_path:str, it=0, figpath="./figures/cossim_distribution.png",delta=False):
    """
    draw the distribution of cosine similarity values of texts with different group of images (i.e. clean v.s. adversarial)
    with seaborn.kdeplot
    
    Parameters:
    :delta: Show distribution of delta values (i.e. the change in cossim) instead of absolute values.
    """
    xaxis = "Δ Cosine Similarity" if delta else "Cosine Similarity"
    # read the csv file
    df = pd.read_csv(csv_path)

    # filter columns with image denoised {it} iterations, clean and adversarial
    clean_img_name_reg = "clean_\\d+_denoised_{:0>3d}times".format(it)
    clean_df = df.filter(regex=clean_img_name_reg)
    # flatten to 1d
    clean_np = clean_df.to_numpy().reshape(-1)
    if delta:
        clean_base = df.filter(regex="clean_\\d+_denoised_000times")
        clean_base = clean_base.to_numpy().reshape(-1)
        clean_np = clean_base-clean_np
    clean_df = pd.DataFrame({xaxis:clean_np})
    print(f"there are {clean_df.shape} clean data points")

    # filter columns with image denoised {it} iterations, clean and adversarial
    adv_img_name_reg = "prompt_constrained_[0-9a-z]+_denoised_{:0>3d}times".format(it)
    adv_df = df.filter(regex=adv_img_name_reg)
    # flatten to 1d
    adv_np = adv_df.to_numpy().reshape(-1)
    if delta:
        adv_base = df.filter(regex="prompt_constrained_[0-9a-z]+_denoised_000times")
        adv_base = adv_base.to_numpy().reshape(-1)
        adv_np = adv_base-adv_np
    adv_df= pd.DataFrame({xaxis:adv_np})
    print(f"there are {adv_df.shape} adv data points")
    #### plot ####
    plt.clf()
    f,ax = plt.subplots(1,1,figsize=(4.3,2))
    sns.kdeplot(clean_df,x=xaxis,color = "#23bac5",ax=ax,fill=True,alpha=0.5,label="clean")
    sns.kdeplot(adv_df,x=xaxis,color = "#fd763f",ax=ax,fill=True,alpha=0.5,label="adversarial")
    plt.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig(os.path.splitext(figpath)[0]+".jpg")
    plt.savefig(os.path.splitext(figpath)[0]+".pdf")

def cossim_line_grouped(csv_path:str, figpath="./figures/cossim_line"):
    """
    plot the line plot of cosine similarity of image after denoising.
    given the path of the similarity matrix csv file, plot the line plot of each image group.
    This function illustrates the trend of cossim during denoise process.
    """
    # read the csv file
    df = pd.read_csv(csv_path)

    # filter columns with image denoised {it} iterations, clean and adversarial
    clean_img_name_reg = "clean_\\d+_denoised_[0-9]{3}times"
    clean_df = df.filter(regex=clean_img_name_reg)
    adv_img_name_reg = "prompt_constrained_[0-9a-z]+_denoised_[0-9]{3}times"
    adv_df = df.filter(regex=adv_img_name_reg)

    clean_img_num = len(set(["_".join(col.split("_")[:-2])+"_" for col in clean_df.columns]))
    adv_img_num = len(set(["_".join(col.split("_")[:-2])+"_" for col in adv_df.columns]))

    # calculate avg and var
    cleanvar = clean_df.var(axis=0)
    # each origin image takes up one row
    cleanvar = cleanvar.to_numpy().reshape((clean_img_num,-1)).mean(axis=0)
    temp = clean_df.mean(axis=0)
    cleanmean = temp.to_numpy().reshape((clean_img_num,-1)).mean(axis=0)
    cleanr1 = list(map(lambda x:x[0]-x[1],zip(cleanmean,cleanvar)))
    cleanr2 = list(map(lambda x:x[0]+x[1],zip(cleanmean,cleanvar)))

    advvar = adv_df.var(axis=0)
    advvar = advvar.to_numpy().reshape((adv_img_num,-1)).mean(axis=0)
    temp = adv_df.mean(axis=0)
    advmean = temp.to_numpy().reshape((adv_img_num,-1)).mean(axis=0)
    advr1 = list(map(lambda x:x[0]-x[1],zip(advmean,advvar)))
    advr2 = list(map(lambda x:x[0]+x[1],zip(advmean,advvar)))


    xticks = [i*50 for i in range(0,cleanmean.shape[-1])]
    plt.figure(figsize=(4.2,2))
    plt.plot(xticks, cleanmean, label="clean",color="#23bac5")
    plt.plot(xticks, advmean, label="adversarial",color="#fd763f")

    # variance
    plt.fill_between(xticks,cleanr1,cleanr2,color="#23bac5",alpha=0.2,edgecolor=None)
    plt.fill_between(xticks,advr1,advr2,color="#fd763f",alpha=0.2,edgecolor=None)
    # plt.legend(bbox_to_anchor=(0.05, 1.05), loc='lower left', borderaxespad=0.,ncol=2)
    plt.legend(ncol=2)
    plt.xlabel("Denoise Tierations")
    plt.ylabel("Cosine Similarity")
    plt.tight_layout()
    plt.savefig(os.path.splitext(figpath)[0]+".png")
    plt.savefig(os.path.splitext(figpath)[0]+".pdf")

def asr_barplot(data = None, name=None,*,width=0.85,OTHER_FONT=13,w=10,h=2.4,uniquename=True):
    if data is None:
        data = {# ["LLaVA-v1.5-7B","InstructBLIP","MiniGPT4-7B","Qwen","GPT4v"]
            "attack w/o defense":(0.609375,0.096875,0.5734375,0.071875,0),
            "CIDER":(0,0.0234375,0.0921875,0.0109375,0),
            "Jailguard":(0.45,0.16,0.195,0.055,0.005)
        }
    xlbl = ["LLaVA-v1.5-7B","InstructBLIP","MiniGPT4","Qwen","GPT4V"]
    xlen = len(data["CIDER"])
    width = width/xlen
    # df = pd.DataFrame.from_dict(data=data,orient="index",columns=["Base","CIDER","Jailguard"])
    # print(df)
    x=np.arange(xlen)
    counter=0
    plt.clf()
    plt.rcParams['font.size'] = 8
    f,ax = plt.subplots(layout="constrained",figsize=(w,h))
    if len(data)==3:
        colors=["#fd763f","#23bac5","#eeca40"]
    elif len(data)==4:
        colors = ["#fd763f","#3b738f","#23bac5","#eeca40"]
    else:
        raise RuntimeError("dismatch color and bar num")
    for k,v in data.items():
        offset = width*counter
        rects = ax.bar(x+offset, v, width, label=k,color=colors[counter], alpha=0.7
                       ) #edgecolor=colors[counter]
        # for r in rects:
        #     if r.get_height()<0.05: # only show value when bar is hardly seen
        # ax.bar_label(rects,padding=1, fmt=lambda x: f'{(x)*100:0.2f}')
        counter+=1
    ax.set_ylabel("Attack Success Rate",fontsize=OTHER_FONT)
    ax.set_xticks(x+width,xlbl[:xlen],
                  rotation=0,fontsize=OTHER_FONT)
    ax.legend(fontsize=OTHER_FONT)
    ax.set_yticklabels([f'{x1*100:.0f}%' for x1 in ax.get_yticks()],fontsize=OTHER_FONT) 
    ax.set_ylim([0,0.7])

    plt.tight_layout()
    if name is None:
        name="asr_bar"
    if uniquename:
        t = int(time.time())
        name=f"{name}_{t}"
    # plt.show()
    plt.savefig(f"./figures/{name}.jpg")
    plt.savefig(f"./figures/{name}.pdf")

def multi_mmvet_barplot(data,groups=["w/o CIDER","CIDER"],colors=["#fd763f","#23bac5"],name="mmvet_detailscore_bar",label_x = -0.4):
    """ 
    Side-by side bar plot on scores for comparison between different methods.  
        data = { #model:origin,defence
        "blip":[[],[]],
        "llava":[],
        "minigpt4":,
        "qwen":
    }"""
    plt.figure()
    f,axs = plt.subplots(2,2,layout="constrained",figsize=(10,4))
    xlbl = ["rec","ocr","know","gen","spat","math","total"]
    OTHER_FONT=13
    for i,key in enumerate(data.keys()):
        ax = axs.flatten()[i]
        total_width, n = 0.8,len(groups)
        width = total_width/n
        x = np.arange(len(xlbl))
        x1 = x-width/2
        x2 = x1+width
        if i==0:
            ax.set_title("\n"+key)
        else:
            ax.set_title(key)
        ax.set_ylabel("Score",fontsize=OTHER_FONT)
        for j in range(len(data[key])):
            xj=x-width*((len(groups)-1)/2 - j)
            rects1 = ax.bar(xj,data[key][j],width=width,label=groups[j],color=colors[j], alpha=0.7)
            # ax.bar_label(rects1,padding=1,fontsize=8,rotation=15)
        # rects2 = ax.bar(x2,data[key][1],width=width,label="CIDer",color="#23bac5")
        # ax.bar_label(rects2,padding=1,fontsize=8,rotation=15)

        ax.set_xticks(x,xlbl,fontsize=OTHER_FONT)
        # ax.set_ylim([0,62])
    # plt.title("\n")
    plt.tight_layout()
    plt.legend(ncol=len(groups),bbox_to_anchor=(label_x,2.73), loc='lower left')
    t = int(time.time())
    plt.savefig(f"./figures/{name}.png")
    plt.savefig(f"./figures/{name}.pdf")

if __name__ == "__main__":
    ## This function could illustrate the difference of clean & adversarial images, by showing a different distribution on delta cosine similarity.
    delta_cos_sim_distribution(csv_path = "output/valset_analysis/simmatrix_valset.csv",it=250,figpath="figures/delta_cossim_dist_valset")

    ## below are part of Figures present in our paper

    ## Figure 3(a)
    clean_adv_cossim_difference(csv_path = "output/valset_analysis/simmatrix_valset.csv")

    ## Figure 3(b)
    grouped_cossim_distribution(csv_path = "output/valset_analysis/simmatrix_valset.csv",it=50)

    ## Figure 3(c)
    cossim_line_grouped(csv_path = "output/valset_analysis/simmatrix_valset.csv")

    ## Figure 3(d)
    grouped_cossim_distribution(csv_path = "output/valset_analysis/simmatrix_valset.csv",it=250,figpath="./figures/delta_cossim_dist",delta=True)

    ## Figure 4 plesae refer to `threshold_selection.plot_tpr_fpr`

    ## Figure 5
    asr_barplot(data = {
        "attack w/o defense":     (0.609375,0.096875,0.5734375,0.071875,0),
        "CIDER":    (0      ,0.0234375,0.0921875,0.0109375,0),
        "Jailguard":(0.45   ,0.16     ,0.195,0.055,0.005)
    },name="cider_asr_bar",uniquename=False,OTHER_FONT=11,width=1.3,w=9)

    ## Figure 6

    multi_mmvet_barplot({
        "InstructBLIP":[[23.5,11.8,17.2,18.8,13.6,11.3,19.4],[14.5,9.4,8.3,8.7,12.3,7.7,12.7]],
        "LLaVA-v1.5-7B":[[34.4,23.2,16.8,19.2,26.8,3.8,31.0],[19.7,18.8,6.7,7.3,21.5,2.5,19.1]],
        "MiniGPT4":[[25.7,15.2,15.6,16.9,18.1,3.8,21.4],[14.9,11.4,8.7,7.5,14.4,3.8,13.0]],
        "Qwen-VL":[[51.9,37.1,41.5,37.9,38.4,19.0,46.2],[29.0,31.5,20.0,17.7,32.2,16.4,29.2]]
    },name="mmvet_newalpha")

    ## Figure 8
    asr_barplot(data = {
        "attack w/o defense":     (0.609375,0.096875,0.5734375,0.071875,0),
        "CIDER-de": (0.25625,0.21875  ,0.43375  ,0.055,0),
        "CIDER":    (0      ,0.0234375,0.0921875,0.0109375,0),
        "Jailguard":(0.45   ,0.16     ,0.195,0.055,0.005)
    },name="ciderde_asr_bar",w=11,width=1.0,uniquename=False,OTHER_FONT=11)

    ## Figure 9
    multi_mmvet_barplot({
        "InstructBLIP":[[23.5,11.8,17.2,18.8,13.6,11.3,19.4],[22.6,10.7,14.9,15.6,12.3,9.0,18.0],[14.5,9.4,8.3,8.7,12.3,7.7,12.7]],
        "LLaVA-v1.5-7B":[[34.4,23.2,16.8,19.2,26.8,3.8,31.0],[31.8,22.0,15.4,18.0,26.7,3.8,28.7],[19.7,18.8,6.7,7.3,21.5,2.5,19.1]],
        "MiniGPT4":[[25.7,15.2,15.6,16.9,18.1,3.8,21.4],[23.1,10.7,12.6,13.5,12.3,2.7,18.1],[14.9,11.4,8.7,7.5,14.4,3.8,13.0]],
        "Qwen-VL":[[51.9,37.1,41.5,37.9,38.4,19.0,46.2],[44.6,34.4,33.9,30.8,37.3,17.7,40.9],[29.0,31.5,20.0,17.7,32.2,16.4,29.2]]
    },name="Fig9-mmvet_ciderde",groups=["attack w/o defense","CIDER-de","CIDER"],colors=["#fd763f","#3b738f","#23bac5"],label_x=-0.61)
