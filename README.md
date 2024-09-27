# Defending Jailbreak Attack in VLMs via Cross-modality Information Detector
<p align="center">
<a href='https://github.com/PandragonXIII/CIDER/blob/main/LICENSE'>
<img src='https://img.shields.io/badge/Code%20License-Apache_2.0-Green'></a> 
<a href='https://arxiv.org/abs/2407.21659'>
<img src='https://img.shields.io/badge/paper-Arxiv-red'></a>
<img src='https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-blue.svg'>
</p>


This is the official repository for ["Cross-modality Information Verification for Defending Multimodal Large Language Models Against Jailbreak Attacks"](https://arxiv.org/abs/2407.21659).

## Abstract
<p align="center">
    <img src="src/workflow.png" width="100%"> <br>
    <br>
    Figure 1: The workflow of safeguarding MLLM against jailbreak attacks via *CIDER*.
</p>

We propose **C**ross-modality **I**nformation **DE**tecto**R** (*CIDER*), a plug-and-play jailbreaking detector designed to identify maliciously perturbed image inputs, utilizing the cross-modal similarity between harmful queries and adversarial images.
This simple yet effective cross-modality information detector, *CIDER*, is independent of the target MLLMs and requires less computation cost. Extensive experimental results demonstrate the effectiveness and efficiency of *CIDER*, as well as its transferability to both white-box and black-box MLLMs.



## Findings

<figure align="center">
  <picture>
    <img src="src/findings.png" width="100%">
  </picture>
  <figcaption>Figure 2: Experimental result. (a) The distribution of the difference between clean and adversarial images regarding their cos-sim with harmful queries. (b) The distribution of cos-sim between harmful queries and clean/adversarial images. (c) The change of the cos-sim during denoising. (d) The distribution of Δcos-sim before and after denoising of clean/adversarial images.</figcaption>
</figure>

1. Adversarial images indeed contain harmful information.
2. Directly utilizing the semantic difference between clean and adversarial images to harmful query is challenging
3. Denoising can reduce harmful information but cannot eliminate
4. Relative shift in the semantic distance before and after denoising can help detect adversarial images.

For detailed explanations, please refer to our [paper](https://arxiv.org/abs/2407.21659).


## CIDER, a plug-and-play cross-modality information detector

*CIDER* is implemented on top of the MLLMs to defense optimization-based adversarial jailbreak attacks. Figure in Abstract presents the overview of the *CIDER* pipeline. 
Specifically, given a text-image input pair, denoted as <*text*, *img(o)*>, *CIDER* calculates the embeddings of text and image modalities, denoted as $\mathbf{E_{\textit{text}}}$ and $\mathbf{E_{\textit{img(o)}}}$. Then, the built-in denoiser in *CIDER* will perform 350 denoising iterations on the image(o), calculating the denoised embeddings every 50 iterations, denoted as $\mathcal{E}=\mathbf{E_{\textit{img(d)}}}$. 
The $\textit{img(o)}$ will be identified as an adversarial example if any $\mathbf{E_{\textit{img(d)}}} \in \mathcal{E}$ satisfy the following condition: 

```math
\begin{align}
\langle \mathbf{E_{\textit{text}}}, \mathbf{E_{\textit{img(o)}}} \rangle - \langle \mathbf{E_{\textit{text}}}, \mathbf{E_{\textit{img(d)}}} \rangle >\tau 
\end{align}
```

where $\langle \cdot \rangle$ represents the cosine similarity and $\tau$ is the predefined threshold. Consequently, *CIDER* will directly refuse to follow the user's request by responding ``I'm sorry, but I can not assist.'' if the image modality is detected as adversarial. Otherwise, the original image and query will be fed into the MLLM. 

### Threshold selection
The threshold is selected based on the harmful queries and clean images ensuring that the vast majority of clean images pass the detection. The selection of threshold $\tau$ can be formulated as: 

```math
\begin{align}
r=\frac{\sum\mathbb{I}(\langle \mathbf{E^\textit{M}_{\textit{text}}}, \mathbf{E^\textit{C}_{\textit{img(o)}}} \rangle - \langle \mathbf{E^\textit{M}_{\textit{text}}}, \mathbf{E^\textit{C}_{\textit{img(d)}}} \rangle <\tau) }{\# \textit{samples}} 
\end{align}
```

where $r$ represents the passing rate and $`\mathbf{E^\textit{M}_{\textit{text}}}`$ , $`\mathbf{E^\textit{C}_{\textit{img(o)}}}`$ , $`\mathbf{E^\textit{C}_{\textit{img(d)}}}`$ stand for the embeddings of input query, the input image and denoised image respectively. The threshold $\tau$ is determined by controlling the passing rate $r$. For example, using the $\tau$ when setting $r$ to 95% as the threshold indicates allowing 95% percent of clean images to pass the detection.
Regarding balance between TPR and FPR, we selected $\tau$ when $r$ equals 95% as the detection threshold of *CIDER*.


## Performance
<p align="center">
    <img src="src/asr.jpg" width="90%"> <br>
    <br>
    Figure 3: ASR of base MLLM, defending with *CIDER* and defending with *Jailguard*
</p>

### Effectiveness
**DSR**: We first demonstrate the overall DSR that *CIDER* can achieve and compare it with the baseline method, *Jailguard*. Table below shows that *CIDER* achieves a DSR of approximately 80%, while the DSR of *Jailguard* varies, depending on the target MLLMs. Note that *CIDER* is independent of the MLLMs, thus the DSR does not vary with the choice of MLLMs. However, *Jailguard*'s detection capability relies heavily on the model's safety alignment, so the DSR also varies. MLLMs with good alignment achieve high DSR (e.g., GPT4V), while poorly aligned MLLMs have relatively low DSR (e.g., InstructBLIP). In other words, *Jailguard* does not significantly enhance MLLM robustness against adversarial jailbreak attacks, whereas *CIDER* does. Nonetheless, *CIDER* achieves a higher DSR than most of the *Jailguard* results, except *Jailguard* on GPT4V.

|   Method   |   detection success rate ($\uparrow$)   |
| ---- | ---- |
|   *Jailguard* with LLaVA-v1.5-7B   |   39.50%   |
|   *Jailguard* with InstructBLIP   |   32.25%   |
|   *Jailguard* with MiniGPT4   |   69.50%   |
|   *Jailguard* with Qwen-VL   |   77.50%   |
|   *Jailguard* with GPT4V   |   94.00%   |
|   **CIDER**   |   79.69%   |


**ASR**: To evaluate the effectiveness of *CIDER*, we measure the decline in ASR after applying *CIDER*. Figure 3 compares the original ASR without defense (red bar), ASR after *CIDER* (blue bar), and ASR after *Jailguard* (yellow bar). Note that, *Jailguard* is solely designed to detect jailbreak input. To ensure a fair comparison, we add an output module following *Jailguard*'s detection. Specifically, if *Jailguard* detects a jailbreak, it will refuse to respond, similar to *CIDER*. Otherwise, the MLLM will process the original input.

Across all models, defending with *CIDER* significantly reduces the ASR, yielding better results than the baseline. This indicates that *CIDER* effectively enhances the robustness of MLLMs against optimization-based jailbreak attacks. The most notable improvements are seen in LLaVA-v1.5-7B, where ASR drops from 60% to 0%, and in MiniGPT4, from 57% to 9%. For MLLMs with initially low ASRs, such as InstructBLIP and Qwen-VL, ASR is reduced to approximately 2% and 1% respectively. Another notable disadvantage of *Jailguard* is observed in models like GPT4V, InstructBLIP, and Qwen-VL, which already had strong safety alignment and resistance to adversarial attacks. In these cases, the use of *Jailguard* resulted in a slight increase in ASR.

We conclude that the threshold determined by *CIDER* can be effectively applied to different MLLMs due to their shared transformer-based LLM backbones, which generate comparable representations of harmful information. This harmful information, distilled from malicious queries, is embedded into adversarial images using similar optimization-based attacks. As a result, the consistent noise patterns produced by these attacks across different MLLMs can be detected using the same threshold, highlighting the robustness and transferability of *CIDER*.


### Efficiency
Timely inference is crucial for safeguarding MLLMs in real-world applications. The table below shows the time required to process a single input pair and generate up to 300 tokens with different MLLMs, comparing no defense, *CIDER*, and *Jailguard*.

| Model | Original | *CIDER* | *Jailguard* |
| ----- | -------- | ------- | ----------- |
|LLaVA-v1.5-7B | $6.39s$ |  $7.41s$ ($1.13\times$) | $53.21s$ ($8.32\times$)|
|InstructBLIP | $5.46s$ |  $6.48s$ ($1.22\times$) | $47.83s$ ($8.76\times$)|
|MiniGPT4 | $37.00s$ |  $38.02s$ ($1.03\times$) | $313.78s$ ($8.48\times$)|
|Qwen-VL | $6.02s$ |  $7.04s$ ($1.19\times$) | $48.48s$ ($8.05\times$)|
|GPT4V| $7.55s$ |  $8.57s$ ($1.16\times$) | $61.04s$ ($8.08\times$)|
<p align="center">
    Time cost to process a single pair of inputs.
</p>


*CIDER* surpasses *Jailguard* in efficiency, adding only 1.02 seconds per input pair on average, which is relatively acceptable compared to the original inference time. In contrast, *Jailguard* requires 8-9 times the original processing time. Additionally, *CIDER* detection is irrelevant to the number of generated tokens in the query answers. Therefore, *CIDER* does not cause additional overhead when increasing the number of generated tokens, ensuring the stability of *CIDER*'s efficiency.

<p align="center">
    <img src="src/mmvet.png" width="100%"> <br>
    <br>
    Figure 4: MLLM performance with and without *CIDER* on MM-Vet.
</p>

To further demonstrate *CIDER*'s influence on the original utilities on normal queries, we also evaluate the utility of *CIDER* protected MLLMs on MM-Vet benchmark, including recognition, OCR, knowledge, language generation, spatial awareness, and math. As shown in Figure 4, employing *CIDER* leads to an approximate 30% overall performance decline on normal tasks. Specifically, *CIDER* mostly affects the MLLM's recognition, knowledge, and language generation capabilities, while it has minimal impact on OCR, spatial awareness, and math skills. We hypothesize that *CIDER*'s stringent decision-making process, which outright rejects tasks once an image is identified as adversarial, hampers the model's overall performance. An ablation study is also included in the paper.


### Open-Source List
We will soon open-source our codes and datasets, stay tuned!


## Acknowledgements

- We thank all reviewers for their constructive comments. 
- This work is supported by the Shanghai Engineering Research Center of Intelligent Vision and Imaging and the Open Research Fund of The State Key Laboratory of Blockchain and Data Security, Zhejiang University.

## Citation

If you find CIDER useful for your research and applications, please cite using the Bibtex:

```bibtex
@inproceedings{
	anonymous2024defending,
	title={Defending Jailbreak Attack in {VLM}s via Cross-modality Information Detector},
	author={Anonymous},
	booktitle={The 2024 Conference on Empirical Methods in Natural Language Processing},
	year={2024},
	url={https://openreview.net/forum?id=pBKvvyg7oM}
}
```