<div align="center">
  
# NeighborRetr: Balancing Hub Centrality in Cross-Modal Retrieval (CVPR'2025 🔥)
  
[![Conference](https://img.shields.io/badge/CVPR-2025-FFD93D.svg)](https://cvpr.thecvf.com/Conferences/2025)
[![Project](https://img.shields.io/badge/Project-NeighborRetr-4D96FF.svg)](https://github.com/zzezze/NeighborRetr)
[![Paper](https://img.shields.io/badge/Paper-arxiv.2503.10526-FF6B6B.svg)](https://arxiv.org/abs/2503.10526)
[![Stars](https://img.shields.io/github/stars/zzezze/NeighborRetr?style=social)](https://github.com/zzezze/NeighborRetr)
</div>

The official implementation of **CVPR 2025** paper: [NeighborRetr: Balancing Hub Centrality in Cross-Modal Retrieval](https://arxiv.org/abs/2503.10526).

> **TL;DR:** *NeighborRetr tackles the hubness problem in cross-modal retrieval by distinguishing between good hubs (relevant) and bad hubs (irrelevant) during training, offering a direct solution rather than relying on post-processing methods that require prior data distributions.*

## 🌟 Overview

The **hubness problem** in cross-modal retrieval refers to the phenomenon where certain items (hubs) frequently emerge as the nearest neighbors to many other samples, while the majority of samples rarely appear as neighbors. This leads to biased representations and degraded retrieval accuracy. Unlike previous approaches that apply post-hoc normalization techniques during inference, NeighborRetr introduces a novel approach that:

- Distinguishes between **good hubs** (semantically relevant) and **bad hubs** (semantically irrelevant)
- Applies adaptive neighborhood adjustment during training
- Employs uniform regularization to balance hub formation


<div align=center>
<img src="static/images/Head.png" width="800px">
</div>

## 📌 Citation
If you find this paper useful, please consider starring 🌟 this repo and citing 📑 our paper:
```bibtex
@article{lin2025neighborretr,
  title={NeighborRetr: Balancing Hub Centrality in Cross-Modal Retrieval},
  author={Lin, Zengrong and Wang, Zheng and Qian, Tianwen and Mu, Pan and Chan, Sixian and Bai, Cong},
  journal={arXiv preprint arXiv:2503.10526},
  year={2025}
}
```

## 😍 Visualization

Our method significantly improves the quality of nearest neighbors, reducing irrelevant hubs and promoting more meaningful semantic relationships:

<div align=center>
<img src="static/images/Visualization.png" width="900px">
</div>

## 🔄 Updates
* **[2025/04/13]**: Code released! 🎉
* **[2025/03/14]**: Initial version submitted to arXiv.
* **[2025/02/27]**: Our paper is accepted to CVPR 2025! 


## 🚀 Quick Start
### Setup

#### Environment Setup
```bash
# Create and activate conda environment
conda create -n NeighborRetr python=3.8 -y
conda activate NeighborRetr

# Install dependencies
pip install -r requirements.txt
```

#### Download CLIP Model
```bash
cd NeighborRetr/models
wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
# Optional: for ViT-B-16
# wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

#### Download Datasets

<div align=center>

|Datasets|Baidu Yun|
|:--------:|:-----------:|
| MSR-VTT | [Download](https://pan.baidu.com/s/1B4bdvuArWsi46Eh0RbvSQw?pwd=5bqe) |
| MSVD | [Download](https://pan.baidu.com/s/1jXR_ySM2u8hlCRRtYXUpQw?pwd=u7mw) |
| ActivityNet | [Download](https://pan.baidu.com/s/1sk92g9Fn0DCbMtYcsWEOEQ?pwd=uyag) |
| DiDeMo | [Download](https://pan.baidu.com/s/1v7OPtuau1zv69FXBAGqb8Q?pwd=puct) |

</div>

### Training

#### Train on MSR-VTT
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch \
--master_port 4501 \
--nproc_per_node=4 \
main_retrieval.py \
--do_train 1 \
--workers 8 \
--epochs 5 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path ${ANNO_PATH} \
--video_path ${VIDEO_PATH} \
--datatype msrvtt \
--max_words 24 \
--max_frames 12 \
--output_dir ${OUTPUT_PATH} \
--mb_batch 15 \
--memory_size 512
```

#### Train on ActivityNet Captions
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
--master_port 4501 \
--nproc_per_node=8 \
main_retrieval.py \
--do_train 1 \
--workers 8 \
--epochs 10 \
--batch_size 128 \
--batch_size_val 128 \
--anno_path ${ANNO_PATH} \
--video_path ${VIDEO_PATH} \
--datatype activity \
--max_words 64 \
--max_frames 64 \
--output_dir ${OUTPUT_PATH} \
--mb_batch 15 \
--memory_size 1024
```

## 📚 License

This repository is released under the [Apache License 2.0](LICENSE). This permissive license allows users to freely use, modify, distribute, and sublicense the code while maintaining copyright and license notices.

## ✨ Acknowledgement

Our work is primarily built upon [HBI](https://github.com/jpthu17/HBI), [CLIP](https://github.com/openai/CLIP), [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip). We extend our gratitude to all these authors for their generously open-sourced code and their significant contributions to the community.
