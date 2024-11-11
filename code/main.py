"""
This file works as an Integrated Block ahead of LLMs to filter potential 
adversarial and pass harmless image with origin text to VLM

input fromat:
- args single csv file with each line containing args text input
    - without head lines
    - Hrambench data is used in here.
- args folder containing input image(s)

output:
- n bmp images under `/output` with same file name(s)
- add args column "input_adv_img" to the csv file denoting classification result
"""

import argparse,os
import pandas as pd
from PIL import Image
import time
import json
import logging

from utils import evaluate_response, defence, QApair, get_response



if __name__=="__main__":
    t0 = time.time()
    parser = argparse.ArgumentParser(description='detect adversarial images and pass harmless images to LLMs')
    parser.add_argument('--text_file', type=str, required=True, help='path to the text input file')
    parser.add_argument('--img', type=str, required=True, help='path to the image input folder or file')
    parser.add_argument('--threshold', type=float, required=True, default=-0.003936767578125, help='threshold for adversarial detection')
    parser.add_argument('--model', type=str, default="llava", help='model to use for generation')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--pair_mode', type=str, default="combine", help='relation between imgs and texts, should be "combine" or "injection"\n combine will create |img|*|text| number of img-text pairs\n injection requires |img|==|text|')
    parser.add_argument('--multirun', type=int, default=1, help="run multiple times")
    parser.add_argument('--outdir', type=str, default="./output", help="dir to place the output file")
    parser.add_argument('--tempdir', type=str, default="./temp", help="dir to place the temporary file")
    parser.add_argument("--denoiser",choices=["diffusion","dncnn"],help="model used for image denoising")
    parser.add_argument('--no_eval', action="store_true", help="disable harmbench evaluation phase on LLM outputs." )
    parser.add_argument('--no_defence', action="store_true", help="disable CIDER detection")
    args = parser.parse_args()

    # create output dir
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    else:
        os.system(f"rm -rf {args.outdir}/*")

    # initialize logger
    logging.basicConfig(level = logging.INFO,filename=f"{args.outdir}/experiment.log",filemode="w",format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    for _ in range(args.multirun):
        # create temp dir
        if not os.path.exists(args.tempdir):
            os.makedirs(args.tempdir)
        else:
            os.system(f"rm -rf {args.tempdir}/*")
        args.image_dir = args.tempdir+"/denoised_img"
        os.mkdir(args.image_dir)
        

        #### read the prompts from csv####
        df = pd.read_csv(args.text_file,header=None)
        # the first column is text inputs
        texts = df[0].tolist()
        text_num = len(texts)
        try:
            behaviours = df[df.shape[1]-1].tolist() # read behave for harmbench eval(v1_x for mmvet)
        except:
            logger.warning(f"no behaviour data found in {args.text_file}")
            behaviours = [i for i in range(df.shape[0])]


        #### read the images ####
        dir1 = os.listdir(args.img)
        dir1.sort()
        image_paths = []
        for f in dir1:
            p = f"{args.img}/{f}"
            if os.path.isfile(p):
                image_paths.append(p)
        # count the number of images
        image_num = len(image_paths)

        # generate pairs
        pairs = [] # type: list[QApair]
        if args.pair_mode=="injection":
            logger.debug(f"imgs initial: {image_paths}")
            if not args.no_defence: # with defence (otherwise use raw imgs)
                denoised_image_paths, refuse = defence(imgpth=image_paths, args=args)
            else:
                denoised_image_paths = image_paths
                refuse = [False]*image_num
            logger.debug(f"imgs denoised: {denoised_image_paths}")
            assert text_num==image_num, "image and text numbers do not match"
            for i in range(image_num):
                pairs.append(
                    QApair(texts[i],denoised_image_paths[i],refuse[i],ans=None,behav=behaviours[i]))
        else: #combine
            if not args.no_defence: # with defence (otherwise use raw imgs)
                denoised_image_paths, refuse = defence(imgpth=image_paths, args=args)
            else:
                denoised_image_paths = [image_paths for _ in range(text_num)]
                refuse = [[False]*image_num]*text_num

            logger.debug(f"img list:{denoised_image_paths}")
            for i in range(text_num):
                for j in range(image_num):
                    pairs.append(
                        QApair(texts[i],denoised_image_paths[i][j],refuse[i][j],ans=None,behav=behaviours[i]))
        
        logger.info(f"Number of Query-Image pairs: {len(pairs)}")

        t_finish = time.time()
        logger.info(f"processed {image_num} images, {text_num} texts in {t_finish-t0:.2f}s")


        # generate responses
        logger.info(f"using {args.model} for generation")
        respons_pairs = get_response(args.model, pairs, device=args.cuda)

        # init dict in the first loop
        if _==0:
            res = {}
            for behav in set(behaviours): # init each behaviour class
                res[behav]=[]
        # save QA pairs to dict
        for p in respons_pairs:
            res[p.behav].append(
                {"test_case":[p.imgpth,p.query],"generation":p.ans}
                )
    
    # save dict to json
    save_path = args.outdir+"/response.json"
    with open(save_path,"w") as f:
        json.dump(res,f)
        logger.info(f"result saved at {save_path}")

    # remove temp file directory
    # os.system(f"rm -rf {args.tempdir}")
    
    t_generate = time.time()
    logger.info(f"Full generation completed in {t_generate-t0:.2f}s")
    if not args.no_eval:
        # evaluate
        logger.info("evaluating...")
        evaluate_response(args.outdir+"/response.json",args.outdir,device=args.cuda)
    logger.info("Attack successfully terminated.")