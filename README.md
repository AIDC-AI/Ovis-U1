<p align="center">
<img src=https://cdn-uploads.huggingface.co/production/uploads/637aebed7ce76c3b834cea37/3IK823BZ8w-mz_QfeYkDn.png width="30%"/></p>

<h1 align="center">
Ovis-U1: Unified Understanding, Generation, and Editing
</h1>

<p align="center">
  <!-- <a href="https://arxiv.org/abs/2502.12579"><img src="https://img.shields.io/badge/arXiv%20paper-2502.12579-b31b1b.svg" alt="arxiv"></a> -->
  <a href="https://github.com/AIDC-AI/Ovis-U1/blob/main/docs/Ovis_U1_Report.pdf"><img src="https://img.shields.io/badge/Paper-Tech_Report-b31b1b" alt="paper"></a>
  <a href="https://github.com/AIDC-AI/Ovis"><img src="https://img.shields.io/badge/GitHub-AIDC--AI/Ovis--U1-blue?style=flat&logo=github" alt="code"></a>
  <a href="https://huggingface.co/spaces/AIDC-AI/Ovis-U1-3B"><img src="https://img.shields.io/badge/🎨_HF_Spaces-AIDC--AI/Ovis--U1--3B-lightblack" alt="demo"></a>
  <a href="https://huggingface.co/AIDC-AI/Ovis-U1-3B"><img src="https://img.shields.io/badge/🤗_Model-AIDC--AI/Ovis--U1--3B-yellow" alt="model"></a>
</p>

<p align="left">
Building on the foundation of the Ovis series, Ovis-U1 is a 3-billion-parameter unified model that  seamlessly integrates <b>multimodal understanding</b>, <b>text-to-image generation</b>, and <b>image editing</b> within a single powerful framework. 
</p>

<p align="center">
  <img src="docs/imgs/Ovis-U1.jpg" width="95%">
  <br>
  <em>The overall architecture of Ovis-U1 (cf. Fig.2 in our report).</em>
</p>

## 🏆 Highlights

*   **Unified Capabilities**: A single model excels at three core tasks: understanding complex scenes, generating images from text, and performing precise edits based on instructions.
*   **Advanced Architecture**: Ovis-U1 features a powerful diffusion-based visual decoder (MMDiT) and a bidirectional token refiner, enabling high-fidelity image synthesis and enhanced interaction between text and vision.
*   **Synergistic Unified Training**: Unlike models trained on single tasks, Ovis-U1 is trained on a diverse mix of understanding, generation, and editing data simultaneously. Our findings show that this approach achieves improved generalization, seamlessly handling real-world multimodal challenges with high accuracy.
*   **State-of-the-Art Performance**: Ovis-U1 achieves leading scores on multiple academic benchmarks, surpassing strong contemporary models in multimodal understanding (69.6 on OpenCompass), generation (83.72 on DPG-Bench), and editing (4.00 on ImgEdit-Bench).


## ✨ Showcase

Here are some examples demonstrating the capabilities of Ovis-U1.

<figure>
  <img src="docs/imgs/examples.png" alt="Ovis-U1 examples">
  <figcaption style="text-align: center;"></figcaption>
</figure>


## 🚀 News

- [2025/6/28] 🔥 Announcing Ovis-U1-3B ([Model](https://huggingface.co/AIDC-AI/Ovis-U1-3B), [Demo](https://huggingface.co/spaces/AIDC-AI/Ovis-U1-3B))!


## 📦 Installation

Ovis-U1 has been tested with Python 3.10, Torch 2.4.0, Transformers 4.51.3, and DeepSpeed 0.15.4. For a full list of package dependencies, please see `requirements.txt`.

```bash
git clone git@github.com:AIDC-AI/Ovis-U1.git
conda create -n ovis-u1 python=3.10 -y
conda activate ovis-u1
cd Ovis-U1
pip install -r requirements.txt
pip install -e .
```

## 🛠️ Inference

We provide simple scripts to test the different capabilities of Ovis-U1.

For multimodal understanding, please run

```bash
python test_img_to_txt.py
```

For text-to-image, please run
```bash
python test_txt_to_img.py \
    --height 1024 \
    --width 1024  \
    --steps 50 \
    --seed 42 \
    --txt_cfg 5  
```

For image editing, please run
```bash
python test_img_edit.py \
    --steps 50 \
    --img_cfg 4 \
    --txt_cfg 7.5  
```

Alternatively, you can try Ovis-U1 directly in your browser on [![Hugging Face Space](https://img.shields.io/badge/🎨_HF_Spaces-AIDC--AI/Ovis--U1--3B-lightblack)](https://huggingface.co/spaces/AIDC-AI/Ovis-U1-3B)


## 📊 Performance

#### OpenCompass Multi-modal Academic Benchmarks

| Model | Avg | MMB | MMS | MMMU | MathVista | Hallusion | AI2D | OCRBench | MMVet | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-4o | **75.4** | **86**  |**70.2** | **72.9** | **71.6** | **57** | **86.3** | 82.2 | **76.9** | 
| InternVL2.5-2B | 59.9 | 70.9 | 54.3 | 43.2 | 51.1 | 42.3 | 74.9 | 80.2 | 62.6 | 
| SAIL-VL-2B | 61 | 73.7 |56.5 | 44.1 | 62.8 | 45.9 | 77.4 | 83.1 | 44.2 | 
| InternVL3-2B | 61.1 | 78 |61.1 | 48.7 | 57.6 | 41.9 | 78.6 | 83.1 | <ins>67</ins> | 
| Qwen2.5-VL-3B | 64.5 | 76.8 | 56.3 | 51.2 | 61.2 | 46.6 | 81.4 | 82.8 | 60 | 
| Ovis2-2B | 65.2 | 76.9 | 56.7 | 45.6 | 64.1 | 50.2 | 82.7 | 87.3 | 58.3 | 
| SAIL-VL-1.5-2B | 67  | 78.5 | 62.6 | 46.4 | 67 | 50 | 83.7 | **89.1** | 58.8 | 
| Ristretto-3B | 67.7 | <ins>80.2</ins> | <ins>62.8</ins> | <ins>51.3</ins> | 67.6 | 50.2 | 84.2 | 84.7 | 60.7 | 
| Ovis-U1 |  <ins>69.6</ins>  | 77.8 |61.3 | 51.1 | <ins>69.4</ins> | <ins>56.3</ins> | <ins>85.6</ins> |  <ins>88.3</ins> | 66.7 | 

#### GenEval

| Model | Overall |Single object | Two object | Counting | Colors | Position | Attribute binding | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-4o | 0.84 | <ins>0.99</ins> | 0.92 | <ins>0.85</ins> | 0.92 | 0.75 | 0.61 | 
| BAGEL | 0.82  | <ins>0.99</ins> | 0.94 | 0.81 | 0.88 | 0.64 | 0.63 | 
| BAGEL 📝 | <ins>0.88</ins> | 0.98 | 0.95 | 0.84 | <ins>0.95</ins> | <ins>0.78</ins> | **0.77** |
| UniWorld-V1 | 0.80 | <ins>0.99</ins> | 0.93 | 0.79 | 0.89 | 0.49 | 0.70 |
| UniWorld-V1 📝 | 0.84 | 0.98 | 0.93 | 0.81 | 0.89 | 0.74 | 0.71 | 
| OmniGen | 0.68 |  0.98 | 0.84 | 0.66 | 0.74 | 0.40 | 0.43 | 
| OmniGen2 |0.80 |  **1** | 0.95 | 0.64 | 0.88 | 0.55 | <ins>0.76</ins> | 
| OmniGen2 📝 | 0.86 | <ins>0.99</ins> | <ins>0.96</ins> | 0.74 | **0.98** | 0.71 | 0.75 | 
| Ovis-U1 |**0.89** |  0.98 | **0.98** | **0.90** | 0.92 | **0.79** | 0.75 | 

*📝 denotes using the rewritten prompts*

#### DPG-Bench

| Model | Overall | Global | Entity | Attribute | Relation | Other | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| BAGEL | **85.07** | **88.94** | **90.37** | **91.29** | <ins>90.82</ins> | <ins>88.67</ins> | 
| UniWorld-V1 |81.38 |  83.64 | 88.39 | 88.44 | 89.27 | 87.22 | 
| OmniGen |81.16 | 87.90 | 88.97 | 88.47 | 87.95 | 83.56 | 
| OmniGen2 |83.57 | <ins>88.81</ins> | 88.83 | <ins>90.18</ins> | 89.37 | **90.27** | 
| Ovis-U1 | <ins>83.72</ins> | 82.37 | <ins>90.08</ins> | 88.68 | **93.35** | 85.20 |

#### ImgEdit-Bench

| Model | Overall |Add | Adjust | Extract | Replace | Remove | Background | Style | Hybrid | Action | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-4o | **4.2** | **4.61** | **4.33** | <ins>2.9</ins> | <ins>4.35</ins> | <ins>3.66</ins> | **4.57** | **4.93** | **3.96** | **4.89** | 
| MagicBrush | 1.90 | 2.84 | 1.58 | 1.51 | 1.97 | 1.58 | 1.75 | 2.38 | 1.62 | 1.22 | 
| Instruct-P2P | 1.88 | 2.45 | 1.83 | 1.44 | 2.01 | 1.50 | 1.44 | 3.55 | 1.2 | 1.46 | 
| AnyEdit | 2.45 | 3.18 | 2.95 | 1.88 | 2.47 | 2.23 | 2.24 | 2.85 | 1.56 | 2.65 | 
| UltraEdit |2.7 | 3.44 | 2.81 | 2.13 | 2.96 | 1.45 | 2.83 | 3.76 | 1.91 | 2.98 | 
| OmniGen |  2.96 | 3.47 | 3.04 | 1.71 | 2.94 | 2.43 | 3.21 | 4.19 | 2.24 | 3.38 |
| Step1X-Edit |3.06 |  3.88 | 3.14 | 1.76 | 3.40 | 2.41 | 3.16 | 4.63 | 2.64 | 2.52 | 
| ICEdit |3.05 | 3.58 | 3.39 | 1.73 | 3.15 | 2.93 | 3.08 | 3.84 | 2.04 | 3.68 | 
| BAGEL |3.2 | 3.56 | 3.31 | 1.7 | 3.3 | 2.62 | 3.24 | 4.49 | 2.38 | 4.17 | 
| UniWorld-V1 |3.26 | 3.82 | 3.64 | 2.27 | 3.47 | 3.24 | 2.99 | 4.21 | 2.96 | 2.74 | 
| OmniGen2 | 3.44 | 3.57 | 3.06 | 1.77 | 3.74 | 3.2 | 3.57 | <ins>4.81</ins> | 2.52 | <ins>4.68</ins> |
| Ovis-U1 |<ins>4.00</ins> | <ins>4.13</ins> | <ins>3.62</ins> | **2.98** | **4.45** | **4.06** | <ins>4.22</ins> | 4.69 | <ins>3.45</ins> | 4.61 | 


#### GEdit-Bench-EN

|  Model | Avg | Background Change | Color Alteration   | Material Modification  | Motion Change | Portrait Beautification  | Style Transfer  | Subject Addition  | Subject Removal  | Subject Replacement  | Text Modification  | Tone Transformation  | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-4o |**7.534** | 7.205 |	6.491 |	**6.607** | **8.096** |	**7.768** |	<ins>6.961</ins> |	7.622 |	**8.331** |	**8.067** |	**7.427** |	**8.301** |	
| AnyEdit | 3.212 | 4.663	| 4.260 |	2.537 |	2.024 |	3.479	| 2.032 |	3.995 |	3.089 |	3.180 |	0.922 |	5.151 |	
| Instruct-Pix2Pix | 	3.684 | 3.825 |	5.182 |	3.688 |	3.509 |	4.339 |	4.560 |	3.461 |	2.031 |	4.237 |	0.955 |	4.733 |
| MagicBrush |4.518 |	5.637 |	5.136 |	5.078 |	4.513 |	4.487 |	4.439 |	5.252 |	3.704 |	4.941 |	1.384 |	5.130 |	
| OmniGen | 5.062 | 5.281 |	6.003 |	5.308 |	2.916 |	3.087 |	4.903 |	6.628 |	6.352 |	5.616 |	4.519 |	5.064 |	
| Gemini |6.315 | 	6.781 |	6.369 |	6.040 |	6.938 |	5.591 |	4.676 |	7.501 |	6.447 |	7.003 |	5.765 |	6.350 |	
| Step1X-Edit |	6.701 | 6.547 |	6.545 |	6.204 |	6.483 |	6.787 |	**7.221** |	6.975 |	6.512 |	7.068 |	<ins>6.921</ins> |	6.448 |	
| Doubao |<ins>6.754</ins> | 	<ins>7.430</ins> |	**7.095** |	6.339 |	<ins>6.973</ins> |	<ins>6.972</ins> |	6.767 |	<ins>7.674</ins> |	6.748 |	<ins>7.447</ins> |	3.471 |	<ins>7.383</ins> |	
| BAGEL | 6.519 | 7.324 |	<ins>6.909</ins> |	<ins>6.381</ins> |	4.753 |	4.573 |	6.150 |	**7.896** |	7.164 |	7.021 |	7.320 |	6.218 |	
| Ovis-U1 |6.420 | **7.486** |	6.879 |	6.208 |	4.790 |	5.981 |	6.463 |	7.491 |	<ins>7.254</ins> |	7.266 |	4.482 |	6.314 |	


## 📚 Citation

If you find Ovis-U1 useful for your research or applications, please cite our technical report:

```bibtex
@inproceedings{wang2025ovisu1,
title={Ovis-U1 Technical Report},
author={Ovis Team},
year={2025}
}
```

## 🙏 Acknowledgments

The code is built upon [Ovis](https://github.com/AIDC-AI/Ovis) and [FLUX](https://github.com/black-forest-labs/flux). We thank their authors for open-sourcing their great work.

## 📄 License

This project is released under Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0, SPDX-License-identifier: Apache-2.0).

## 🚨 Disclaimer

We used compliance checking algorithms during the training process, to ensure the compliance of the trained model to the best of our ability. Due to complex data and the diversity of language model usage scenarios, we cannot guarantee that the model is completely free of copyright issues or improper content. If you believe anything infringes on your rights or generates improper content, please contact us, and we will promptly address the matter.
