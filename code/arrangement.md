## needed functions in main:
- [x] defence
- [x] get_response
- [x] evaluate_response: impl in utils


# tests to run
- [ ] models:
    - [x] minigpt4
    - [x] qwen
    - [x] blip 
    - [ ] gpt4
- [x] with defence
- [x] w/o defence
- [x] with eval
- [x] denoiser: DnCNN
- [x] threshold selection
- [x] injecction


还需要整理的：
- [x] ~~get_embed.py~~ moved to `threshold_selection.py`
- [x] defender.py
- [x] model_tools.py
- [x] model.py
- [x] visualization.py
- [x] threshold_selection.py(main部分)

~~settings 里的路径改掉~~
~~code/models/minigpt4/configs/models/minigpt4_vicuna0.yaml line 18 路径~~
~~code/models/minigpt4/minigpt4_eval.yaml 路径~~
~~README的title和链接名字记得改~~
~~requirement整理好~~


you mey need to download the following models:
for minigpt4, you need:
- [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
for Qwen-VL, you need:
- [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat)