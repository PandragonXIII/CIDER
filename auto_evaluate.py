"""
automatically test all models with/without defence using testset/mm-vet dataset
use `python3 ./autoattack.py` to run this script.
"""
import os,json

def reformat(filename:str,read_dir:str,save_dir:str):
    """
    read data from read_dir and save the reformated to save_dir for mmvst test
    filename: saved file name
    the file to read is default as response.json
    """
    ndic = dict()
    with open(f"{read_dir}/response.json") as f:
        dic = json.load(f)
        print(f"query number: {len(dic.keys())}")
        for k,v in dic.items():
            ndic[k]=[conversation["generation"] for conversation in v]
    dp = os.path.join(save_dir,filename.split(".")[0])
    if not os.path.exists(dp):
        os.mkdir(dp)
    
    with open(f"{dp}/{filename}","w") as f1:
        json.dump(ndic, f1)
    # save original response json
    with open(f"{dp}/response.json","w") as f:
        json.dump(dic,f)
    if os.path.exists(f"{read_dir}/response_eval.json"):
        os.system(f"cp {read_dir}/response_eval.json {dp}/")

# imgdir = f"/home/xuyue/QXYtemp/mm-vet-data/images218"
# textfile = f"/home/xuyue/QXYtemp/mm-vet-data/mmvet_query.csv"
imgdir = "data/img/adv4" # test set
textfile = "./data/text/testset.csv" # test set

group_dir = "testset" # subdir name of this run
DEFAULT_THRESHOLD = -0.003936767578125
denoiser = "dncnn"


models = ["blip"]
for i,model in enumerate(models):
    name = f"{model}_{imgdir.split('/')[-1]}_{os.path.splitext(textfile)[0].split('/')[-1]}"
    os.system(f"""python3 ./code/main.py --text_file {textfile} \
    --outdir output/{group_dir}/{name}\
    --img {imgdir} --model {model} \
    --pair_mode combine  --threshold {DEFAULT_THRESHOLD} \
    --cuda 1 --tempdir ./temp/{name} --denoiser {denoiser} &""")
    print (f"{model} generating in process\nsrc files: {textfile}, {imgdir}")