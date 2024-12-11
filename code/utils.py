# Load model directly
import torch
import json,os
import tqdm
import yaml
import logging, functools
import numpy as np
import csv
import time
import types
import shutil  

from math import ceil
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, AutoModelForPreTraining, AutoTokenizer, AutoModelForCausalLM, AutoImageProcessor,Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


from models.diffusion_denoiser.imagenet.DRM import DiffusionRobustModel
from models.DnCNN.DnCNN import DnCNN
from defender import Defender
from models.minigpt4.common.config import Config
from models.minigpt4.common.registry import registry
from models.minigpt4.conversation.conversation import Chat,CONV_VISION_Vicuna0
from openai import OpenAI
import httpx
import base64
from io import BytesIO
import abc


logger = logging.getLogger(__name__)

def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            logger.info(f"running {func.__name__}")
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} DONE.")
            return result
        except Exception as e:
            logger.exception(f"Exception raised in {func.__name__}. exception: {str(e)}")
            raise e
    return wrapper

with open("./settings/settings.yaml") as f:
    settings = yaml.safe_load(f)


LLAMA2_CLS_PROMPT = settings["LLAMA2_CLS_PROMPT"]

@log
def evaluate_response(response_file, outputdir, device="auto"):
    """receive LLM-generated answers and output its harmfulness and ASR"""
    cls = AutoModelForCausalLM.from_pretrained(settings["harmbench_llama_path"], torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(settings["harmbench_llama_path"], use_fast=False, truncation_side="left", padding_side="left")
    
    with open(response_file) as f:
        raw = json.load(f)

    all_responses = []
    for l in tqdm.tqdm(raw.values()):
        tot_generations = [case["generation"] for case in l]
        tot_behavior = [case["test_case"][1] for case in l]
        tot_compl = []
        for generations,behavior in zip(tot_generations,tot_behavior):
            generations = [generations]
            behavior = [behavior]
            context = None
            if context:
                inputs = [LLAMA2_CLS_PROMPT['prompt_contextual'].format(behavior=behavior, generation=g, context=context) for g in generations]
            else:
                inputs = [LLAMA2_CLS_PROMPT['prompt'].format(behavior=behavior, generation=g) for g in generations]

            ###################
            encoded = tokenizer(inputs, return_tensors='pt', padding='longest')
            with torch.no_grad():
                output_ids = cls.generate(
                    **encoded.to(cls.device),
                    do_sample=False,
                    max_new_tokens=1,
                ).cpu()
                output_ids = output_ids[:, len(encoded.input_ids[0]):]

            completion = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
            tot_compl.extend(completion)
        all_responses.extend(tot_compl)# all_responses = ['Yes', 'Yes', 'No', ...]
    y,n=all_responses.count('Yes'),all_responses.count('No')
    # this work in most cases
    if y+n!=len(all_responses): # check if all outputs were counted
        logger.warning(f"There should be {len(all_responses)} results but found {y+n}")
    logger.info(f"\nYes:{y}\nNo:{n}\nASR:{y/(y+n)}")
    res = {
        "YES":y, "NO":n, "ASR": y/(y+n),
        "full_result": all_responses
    }
    write_path = os.path.join(outputdir,"eval.json")
    with open(write_path,"w") as fw:
        json.dump(res,fw)
        logger.info(f"evaluation result saved at {write_path}")

@log
def defence(imgpth:list[str], args, cpnum=8)->tuple[list[str]|list[list], list[bool]|list[list[bool]]]:
    """
    resize the input image if needed
    denoise, calculate cossim and determine which are safe
    
    input: image_paths, args.tempdir
    return: (processed_images_paths, refuse_vector)
        if conbine: shape(processed_images_paths)=|queries|*|imgs|
        otherwise injection: shape(processed_images_paths)=|queries|=|imgs|
        the same is for refuse_vector
    """
    # denoise the image
    logger.info("processing images...")
    # count the number of images in args.img
    image_num = len(imgpth)
    denoised_img_dir = os.path.join(args.tempdir,"denoised_imgs")

    if args.denoiser=="dncnn":
        cpnum=2 # DnCNN currently only denoise once for each img.
    denoised_imgpth = generate_denoised_img(args.denoiser, imgpth=imgpth, save_dir=denoised_img_dir, cps=cpnum, batch_size=50,device=args.cuda, model_path=[settings["DnCNN_path"]])
    denoised_imgpth.sort()
    logger.debug(f"de_imgs:{denoised_imgpth}")

    # compute cosine similarity
    sim_matrix = get_similarity_list(args.text_file,args.pair_mode,imgdir=denoised_img_dir,cpnum=cpnum,device=args.cuda)
    logger.debug(f"shape of sim_matrix: {sim_matrix.shape}")
    # for each row, check with detector
    d_denoise = Defender(threshold=args.threshold)
    adv_idx = []
    refuse = [] # type:list|list[list]
    ret_images_pths = []
    if args.pair_mode=="combine": # 3d-array, text,img,cpnum
        text_num = sim_matrix.shape[0]
        logger.debug(f"sim_matrix[0][0]: {sim_matrix[0][0]}")
        for i in range(text_num):
            adv_row,refuse_row,ret_img_row = [],[],[]
            for j in range(image_num):
                low = d_denoise.get_lowest_idx(sim_matrix[i][j])
                adv_row.append(low)
                refuse_row.append(low!=0)
                # select the lowest img and return
                idx = j*cpnum+low
                ret_img_row.append(denoised_imgpth[idx])
            adv_idx.append(adv_row)
            refuse.append(refuse_row)
            ret_images_pths.append(ret_img_row)
    else: #injeciton
        for i in range(sim_matrix.shape[0]): #=|queries|=|imgs|
            # the defender will find idx of image with lowest cossim(with decrease over threshold)
            adv_idx.append(d_denoise.get_lowest_idx(sim_matrix[i]))
            # and once found, VLM should refuse to respond.
            refuse.append(adv_idx[i]!=0)
            # select the lowest img and return
            idx = i%image_num*cpnum+adv_idx[i] 
            ret_images_pths.append(denoised_imgpth[idx])
        
    return (ret_images_pths, refuse)


##################
class Encoder():
    model_path = None
    def __init__(self, mdpth) -> types.NoneType:
        self.model_path = mdpth
    
    @staticmethod
    def compute_cosine(a_vec:np.ndarray , b_vec:np.ndarray):
        """calculate cosine similarity"""
        norms1 = np.linalg.norm(a_vec, axis=1)
        norms2 = np.linalg.norm(b_vec, axis=1)
        dot_products = np.sum(a_vec * b_vec, axis=1)
        cos_similarities = dot_products / (norms1 * norms2) # ndarray with size=1
        return cos_similarities[0]
    
    @abc.abstractmethod
    def calc_cossim(self,pairs:list[tuple[str,str]]):
        """input list of (query, img path) pairs, 
        output list of cosin similarities"""
        res = []
        for p in pairs:
            text_embed = self.embed_text(p[0])
            img_embed = self.embed_img(p[1])
            cossim = self.compute_cosine(text_embed,img_embed)
            res.append(cossim)
        return res

    @abc.abstractmethod
    def embed_img(self,imgpth)->np.ndarray:
        pass

    @abc.abstractmethod
    def embed_text(self,text)->np.ndarray:
        pass


class QwenEncoder(Encoder):
    min_pixels=224*224
    max_pixels=1024*1024

    def __init__(self,mdpth,device="cuda:0"):
        super().__init__(mdpth)
        self.device=device
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(settings["Qwen2_VL_7B"]).to(device)
        self.visual_model = self.model.visual.to(device)
        self.processor = AutoProcessor.from_pretrained(settings["Qwen2_VL_7B"], min_pixels=self.min_pixels, max_pixels=self.max_pixels)
    
    def embed_img(self,imgpth)->np.ndarray:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": imgpth,
                        'max_pixels': self.max_pixels,
                        'min_pixels': self.min_pixels,
                    },
                    {"type": "text", "text": ""},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        pixel_values = inputs["pixel_values"].type(torch.bfloat16)

        image_embeds = self.visual_model(pixel_values, grid_thw=inputs["image_grid_thw"]) # shape: [n tokens, 3584]

        # calculate average to compress the 2th dimension
        image_features = torch.mean(image_embeds, dim=0).detach().to("cpu").numpy().reshape(1,-1)
        return image_features

    def embed_text(self, text)->np.ndarray:
        text = self.processor.apply_chat_template(text, tokenize=False, add_generation_prompt=True)
        input_ids = self.processor.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        input_embeds = self.model.model.embed_tokens(input_ids)
        # calculate average to get shape[1, 3584]
        input_embeds = torch.mean(input_embeds, dim=1).detach().to("cpu").numpy()
        return input_embeds


class LlavaEncoder(Encoder):
    def __init__(self, mdpth,device="cuda:0") -> types.NoneType:
        super().__init__(mdpth)
        self.device=device
        self.model = AutoModelForPreTraining.from_pretrained(
        mdpth, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(mdpth)
        self.imgprocessor = AutoImageProcessor.from_pretrained(mdpth)

    def embed_img(self, imgpth) -> np.ndarray:
        image = Image.open(imgpth)
        # img embedding
        pixel_value = self.imgprocessor(image, return_tensors="pt").pixel_values.to(self.device)
        image_outputs = self.model.vision_tower(pixel_value, output_hidden_states=True)
        selected_image_feature = image_outputs.hidden_states[self.model.config.vision_feature_layer]
        selected_image_feature = selected_image_feature[:, 1:] # by default
        image_features = self.model.multi_modal_projector(selected_image_feature)
        # calculate average to compress the 2th dimension
        image_features = torch.mean(image_features, dim=1).detach().to("cpu").numpy()
        return image_features

    def embed_text(self, text) -> np.ndarray:
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        input_embeds = self.model.get_input_embeddings()(input_ids)
        # calculate average to get shape[1, 4096]
        input_embeds = torch.mean(input_embeds, dim=1).detach().to("cpu").numpy()
        return input_embeds

@log
def get_similarity_list(text_file,pair_mode, imgdir="./temp/denoised_imgs",device="cuda:0", encoder_pth:str=settings["Embed_model_path"], cpnum = 8):
    """
    calculate the cosine similarity matrix between each text and denoised images and save the result as csv.
    called by main.py

    input:
        :text_file: csv file without headers.
    return: 
    np.ndarray
        :width: number of images in the dir
        :height: len(text_embed_list)
    """
    if "qwen" in encoder_pth.lower():
        encoder = QwenEncoder(encoder_pth,device)
    elif "llava" in encoder_pth.lower():
        encoder = LlavaEncoder(encoder_pth,device)
    else:
        raise ValueError(f"Unrecognised encoder Type from:{encoder_pth}")
    
    # load text inputs
    with open(text_file) as fr:
        reader = csv.reader(fr)
        queries = [line[0] for line in reader if line[1]=="standard"]
    # load image paths
    dir1 = sorted(os.listdir(imgdir))
    img_pths = [os.path.join(imgdir,img) for img in dir1]
    # compute cosine similarity between text and n denoised images
    # and form a table of size ((len(queries),image_num, ckpt_num)) or (len(text_embed_list), ckpt_num)
    image_num = len(img_pths)//cpnum
    if pair_mode=="combine":
        cossims = np.zeros((len(queries),image_num, cpnum))
        for i in range(len(queries)):
            for j in range(image_num):
                inputs = [(queries[i],img_pths[k]) for k in range(j*cpnum,(j+1)*cpnum)]
                temp = encoder.calc_cossim(inputs)
                cossims[i,j] = temp
    else: # injection
        cossims = np.zeros((len(queries), cpnum))
        for i in range(image_num):
            inputs = [(queries[i],img_pths[k]) for k in range(i*cpnum,(i+1)*cpnum)]
            temp = encoder.calc_cossim(inputs)
            cossims[i] = temp
    return cossims



@log
def generate_denoised_img(model="diffusion",**kwargs):
    try:
        shutil.rmtree(kwargs["save_dir"])
    except FileNotFoundError:
        logger.info("No existing temp dir")
    finally:
        os.makedirs(kwargs["save_dir"])
    
    if model=="diffusion":
        func = generate_denoised_img_diffusion
    elif model=="dncnn":
        func = generate_denoised_img_DnCNN
    elif model=="nlm":
        func = generate_denoised_img_NLM
    else:
        raise RuntimeError(f"Unrecognised model type: {model}")
    return func(**kwargs)
@log
def generate_denoised_img_diffusion(imgpth:list[str], save_dir:str, cps:int , step=50, device:int=0, batch_size = 50, **kwargs):
    """
    read all img file under given dir, and convert to RGB
    copy the original size image as denoise000,
    then resize to 224*224,
    denoise for multiple iterations and save them to save_path
    
    :imgpth: list of path of image(s)
    :cps: denoise range(0,step*cps,step) times
            step=50 by default
    """
    
    resized_imgs = [] #type: list[tuple[Image.Image,str]]
    denoised_pth = []
    imgpth.sort()
    
    # preprocess
    trans = transforms.Compose([transforms.Resize([224,224])])
    for filepath in imgpth:
        img = Image.open(filepath).convert("RGB")
        filename = os.path.split(filepath)[1]
        # resize to 224*224
        img = trans(img)
        resized_imgs.append((img,filename))
        # save the original image
        savename=os.path.splitext(filename)[0]+"_denoised_000times.jpg"
        img.save(os.path.join(save_dir,savename))
        denoised_pth.append(os.path.join(save_dir,savename))
    if cps<=1:
        return denoised_pth
    
    model = DiffusionRobustModel(device=f"cuda:{device}")
    iterations = range(step,cps*step,step)
    b_num = ceil(len(resized_imgs)/batch_size) # how man runs do we need
    for b in tqdm.tqdm(range(b_num),desc="denoising batch"):
        l = b*batch_size
        r = (b+1)*batch_size if b<b_num-1 else len(resized_imgs)
        # denoise for each part between l and r
        part = resized_imgs[l:r]
        for it in iterations:
            # project value between -1,1
            ary = [np.array(_[0],dtype=np.float32)/255*2-1 for _ in part]
            ary = np.array(ary)
            ary = torch.tensor(ary).permute(0,3,1,2).to(device)
            denoised_ary=np.array(model.denoise(ary, it).to("cpu"))
            denoised_ary=denoised_ary.transpose(0,2,3,1)
            denoised_ary = (denoised_ary+1)/2*255
            denoised_ary = denoised_ary.astype(np.uint8)
            for i in range(denoised_ary.shape[0]):
                img=Image.fromarray(denoised_ary[i])
                sn = os.path.splitext(part[i][1])[0]+"_denoised_{:0>3d}times.jpg".format(it)
                img.save(os.path.join(save_dir,sn))
                denoised_pth.append(os.path.join(save_dir,sn))
    del model
    torch.cuda.empty_cache()
    return denoised_pth
@log
def generate_denoised_img_NLM(path:str, save_dir, cps:int , step=1, device="cuda:1", batch_size = 50, **kwargs):
    """
    read all img file under given dir, and convert to RGB
    copy the original size image as denoise000,
    then resize to 224*224,
    denoise for multiple iterations and save them to save_path
    using Non Local Means (NLM)
    
    :path: path of dir or file of image(s)
    :cps: denoise range(0,step*cps,step) times
            step=1 by default
    """
    raise NotImplementedError
    assert os.path.isdir(path)
    resized_imgs = []
    img_names = []
    dir1 = os.listdir(path) # list of img names
    dir1.sort()
    for filename in dir1:
        img = Image.open(path+"/"+filename).convert("RGB")
        # save the original image
        savename=os.path.splitext(filename)[0]+"_denoised_000times.jpg"
        img.save(os.path.join(save_path,savename))
        # resize for denoise
        resized_imgs.append(
            resize_keep_ratio(img=img)
        )
        img_names.append(filename)
    if cps<=1:
        return
    
    # model = DiffusionRobustModel(device=int(device[-1]))
    iterations = range(step,cps*step,step)
    b_num = ceil(len(resized_imgs)/batch_size) # how man runs do we need
    for b in tqdm.tqdm(range(b_num),desc="denoise batch"):
        l = b*batch_size
        r = (b+1)*batch_size if b<b_num-1 else len(resized_imgs)
        # denoise for each part between l and r
        part = resized_imgs[l:r]
        partname = img_names[l:r]
        for it in iterations:
            # project value between -1,1
            ary = [np.array(_,dtype=np.float32)/255*2-1 for _ in part]
            ary = np.array(ary)
            denoised_ary = []
            for img_temp in ary:
                sigma_est = np.mean(estimate_sigma(img_temp, channel_axis=-1))
                patch_kw = dict(
                    patch_size=5, patch_distance=6, channel_axis=-1  # 5x5 patches  # 13x13 search area
                )
                for _ in range(it):
                    img_temp = denoise_nl_means(
                        img_temp, h=0.8 * sigma_est, sigma=sigma_est, fast_mode=False, **patch_kw
                    )
                denoised_ary.append(img_temp)
            denoised_ary = np.array(denoised_ary)
            denoised_ary = (denoised_ary+1)/2*255
            denoised_ary = denoised_ary.astype(np.uint8)
            for i in range(denoised_ary.shape[0]):
                img=Image.fromarray(denoised_ary[i])
                sn = os.path.splitext(partname[i])[0]+"_denoised_{:0>3d}times.jpg".format(it)
                img.save(os.path.join(save_path,sn))
    # del model
    torch.cuda.empty_cache()
    return
@log
def generate_denoised_img_DnCNN(imgpth:list[str], save_dir:str, device:int|torch.device, batch_size = 50, model_path=[settings["DnCNN_path"]], **kwargs):
    """
    read all img file under given dir, and convert to RGB
    copy the original size image as denoise000,
    then resize to 224*224,
    denoise with DnCNN and save them to save_path
    
    :path: path of dir or file of image(s)
    """
    denoised_pth = [] # return
    toPilImage = transforms.ToPILImage()
    resized_imgs = []
    img_names = []
    imgpth.sort()
    
    resize = transforms.Compose([transforms.Resize([224,224])])
    totensor = transforms.ToTensor()
    for filepath in imgpth:
        img = Image.open(filepath).convert("RGB")
        img = resize(img)
        filename = os.path.split(filepath)[1]
        # save the original image
        savename=os.path.splitext(filename)[0]+"_denoised_000times.jpg"
        img.save(os.path.join(save_dir,savename))
        denoised_pth.append(os.path.join(save_dir,savename))
        # rconvert to tensor
        img = totensor(img)
        img = torch.unsqueeze(img,0).cuda(device) # type: ignore # 填充一维
        resized_imgs.append(img)
        img_names.append(filename)
    
    # load multi DnCNN nets
    denoisers=[]
    for ckpt in model_path:
        logger.debug(f"loading DnCNN from: {ckpt}")
        checkpoint = torch.load(ckpt)
        denoiser = torch.nn.DataParallel(DnCNN(image_channels=3, depth=17, n_channels=64),device_ids=[device])
        torch.backends.cudnn.benchmark = True
        denoiser.load_state_dict(checkpoint['state_dict'])
        denoiser.cuda(device).eval()
        denoisers.append(denoiser)


    # iterations = range(step,cps*step,step)
    b_num = ceil(len(resized_imgs)/batch_size) # how many runs do we need
    for b in tqdm.tqdm(range(b_num),desc="denoise batch"):
        l = b*batch_size
        r = (b+1)*batch_size if b<b_num-1 else len(resized_imgs)
        # denoise for each part between l and r
        part = resized_imgs[l:r]
        partname = img_names[l:r]
        with torch.no_grad():
            for i,img in enumerate(part):
                for j,d in enumerate(denoisers):
                    outputs = d(img)
                    outputs = torch.clamp(outputs, 0, 1) # remember to clip pixel values
                    denoised = toPilImage(outputs[0].cpu())
                    sn = os.path.splitext(partname[i])[0]+f"_denoised_{j+1}times.jpg"
                    denoised.save(os.path.join(save_dir,sn))
                    denoised_pth.append(os.path.join(save_dir,sn))
    del denoisers
    torch.cuda.empty_cache()
    return denoised_pth

class QApair:
    """
    containing all informations for a query-answer process, including:
    - query
    - image path
    - refuse: bool, if is True(harmful image), VLM should directly refuse to answer.
    - answer
    - behaviour: label in Harmbench, will be used in scoring process.
    """
    def __init__(self, query, imgpth, refuse, ans, behav) -> None:
        self.query = query
        self.imgpth = imgpth
        self.refuse = refuse
        self.ans = ans
        self.behav = behav

try:
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration # llava 1.6
except ImportError as e:
    logger.warning(f"llava1.6 not available, err msg:\n{e}")
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

@log
def get_response(model_name, pairs:list[QApair], device=0)->list[QApair]:
    """
    get different model response with given texts and images pairs
    :model_name: one of "llava", "minigpt4", "blip","gpt4"...
    """
    logger.debug(f"Pairs[0]:{pairs[0].query,pairs[0].imgpth}")
    try: # load openai apikey and url
        openaisettingfile = "../openai_env_var.json"
        openai_settings = json.load(open(openaisettingfile))
        os.environ["OPENAI_API_KEY"] = openai_settings["api_key"]
        os.environ["OPENAI_BASE_URL"] = openai_settings["url"]
    except:
        logger.warning(f"No openai settings were found in {openaisettingfile}, create one if you want to use GPT as target model.")
    t_start=time.time()
    if model_name in ["llava", "blip", "llava1.6"]:
        if model_name=="llava":
            processor = AutoProcessor.from_pretrained(settings["llava15_7b_hf_path"])
            model = AutoModelForPreTraining.from_pretrained(settings["llava15_7b_hf_path"]) 
        elif model_name=="llava1.6":
            try:
                processor = LlavaNextProcessor.from_pretrained(settings["llava16_7b_hf_path"]) # type: ignore
                model = LlavaNextForConditionalGeneration.from_pretrained(settings["llava16_7b_hf_path"]) # type: ignore
            except:
                raise UserWarning("llava1.6 requires different environment settings, please checkout in advance!")
        else: # blip
            processor = InstructBlipProcessor.from_pretrained(settings["instructblip_path"])
            model = InstructBlipForConditionalGeneration.from_pretrained(settings["instructblip_path"])
        model.to(device) # type: ignore
        for i,p in tqdm.tqdm(enumerate(pairs),desc="Generating responses"):
            if p.refuse==True:
                pairs[i].ans="sorry,  I can not assist with that."
                continue
            input = processor(text=f"<image>\n{p.query}\n", images=Image.open(p.imgpth), return_tensors="pt").to(device)
            # autoregressively complete prompt
            output = model.generate(**input,max_new_tokens=512) # type: ignore
            outnpy=output.to("cpu").numpy()
            answer = processor.decode(outnpy[0], skip_special_tokens=True)
            pairs[i].ans = answer.replace(f"\n{p.query}\n","").strip() #remove the query in dialogue and get answer.
        del model
        torch.cuda.empty_cache()
    elif model_name.lower()=="minigpt4":
        # load models for i2t
        # settings of minigpt4
        argdict = types.SimpleNamespace(
            cfg_path="./code/models/minigpt4/minigpt4_eval.yaml",options=None) 
        cfg = Config(argdict)
        # logger.info(f"Loading MiniGPT-4 models from {cfg.config.}")
        cfg.pretty_print()
        model_config = cfg.model_cfg
        model_config.device_8bit = device
        model_cls = registry.get_model_class(model_config.arch)  # model_config.arch: minigpt-4
        model = model_cls.from_config(model_config).to(device)

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor     = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)       
        num_beams = 1
        temperature = 1.0
        chat = Chat(model, vis_processor, device=f"cuda:{device}")
        logger.info("Finished Loading minigpt4")
        # start querying
        for i,p in tqdm.tqdm(enumerate(pairs)):
            if p.refuse==True:
                pairs[i].ans="sorry,  I can not assist with that."
                continue
            pairs[i].ans = query_minigpt(
                question="<image>\n"+p.query, img=Image.open(p.imgpth),chat=chat
            )
        del chat,model,vis_processor
        torch.cuda.empty_cache()
    elif model_name=="qwen":
        tokenizer = AutoTokenizer.from_pretrained(settings["qwen_path"],trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(settings["qwen_path"],trust_remote_code=True,fp32=True).eval()
        model.to(device)
        for i,p in tqdm.tqdm(enumerate(pairs)):
            if p.refuse==True:
                pairs[i].ans="sorry,  I can not assist with that."
                continue
            input = tokenizer.from_list_format([{"image":p.imgpth},{"text":p.query}]) # type: ignore
            # autoregressively complete prompt
            answer, history = model.chat(tokenizer, query=input, history=None ,max_new_tokens=512)
            pairs[i].ans = (answer)
        del model
        torch.cuda.empty_cache()
    elif model_name=="gpt4":
        client = OpenAI(
        http_client=httpx.Client(
            base_url=openai_settings["url"],
            follow_redirects=True,
            ),
        )
        
        for i,p in tqdm.tqdm(enumerate(pairs)):
            if p.refuse==True:
                pairs[i].ans="sorry,  I can not assist with that."
                continue
            base64_image = encode_image(p.imgpth)
            errorcnt=1
            while errorcnt>0:
                try: # api may raise error
                    response = client.chat.completions.create(
                        model="gpt-4-vision-preview",
                        messages=[
                            {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": p.query},
                                {"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}",
                                            "detail": "auto"},
                                }
                            ],
                            }
                        ],
                        max_tokens=300
                    )
                except Exception as e:
                    errorcnt+=1
                    time.sleep(3)
                    if errorcnt>6:
                        logger.error(f"exception encountered with query{i}")
                        logger.error(e)
                        answer=""
                        break
                else:
                    errorcnt=0 # successfully generated, break loop
                    answer = response.choices[0].message.content
            pairs[i].ans = answer
            time.sleep(1)
    else:
        raise Exception("unrecognised model_name, please choose from llava,blip,minigpt4,qwen,gpt4")
    t_gen = time.time()-t_start
    logger.info(f"Generation responses finished in {t_gen:.2f}s.")
    return pairs

# helper functions for miniGPT-4
def upload_img(chat,img):
    CONV_VISION = CONV_VISION_Vicuna0
    chat_state = CONV_VISION.copy()
    img_list = []
    chat.upload_img(img, chat_state, img_list)
    chat.encode_img(img_list)
    return chat_state, img_list

def ask(chat,user_message, chat_state):
    chat.ask(user_message, chat_state)
    return chat_state

def answer(chat,chat_state, img_list, num_beams=1, temperature=1.0):
    llm_message  = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=512,
                              max_length=2000)[0]

    return llm_message, chat_state, img_list

def query_minigpt(question,img,chat):
    with torch.no_grad():
        chat_state, img_list = upload_img(chat,img)
        chat_state = ask(chat,question, chat_state)
        llm_message, chat_state, img_list = answer(chat,chat_state, img_list)
    return llm_message

# helper functions for GPT
# REMINDER: gpt4 only support [png,jp(e)g,webp,gif] at present
def encode_image(image_path):
    """convert image to jpeg and encode with base64"""
    img = Image.open(image_path).convert("RGB")
    im_file = BytesIO()
    img.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    return base64.b64encode(im_bytes).decode('utf-8')