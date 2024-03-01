# IIDRS

Welcome to the official GitHub repository for IIDRS. We apologize for the delay in making this available.

This repository contains the dataset and code implementation utilized in our research paper [here](https://arxiv.org/pdf/2310.07287.pdf). For those interested in exploring or utilizing our work, here's what you can find here:

- **Dataset**: Access the dataset used in our study [here](https://drive.google.com/drive/folders/1I1Jy9m8s7cQq99M8cYuCAyryApjPacnW). This dataset is crucial for replicating our results and furthering research in this domain.

- **Code Implementation**: Dive into our codebase to see how we've implemented the algorithms and models discussed in the paper. You can find the complete code and  file structure[here](https://drive.google.com/drive/folders/1_vi2RwjKn_NAU7vpKhUiPO-HH1I_HEcn).

- **Pre-trained Models**: To facilitate ease of use and quick integration, we've made our trained models available [here](https://drive.google.com/drive/folders/1ERbvDfGaDIiaUhIFmRDOaSFPzDXMpY_B).


## Getting Started

To get started with our dataset and codebase, please ensure you have the necessary prerequisites installed. Here's a quick guide on how to set up your environment:

```bash
# Clone this repository
git clone git@github.com:ZhHe11/IIDRS.git
cd IIDRS

# Install dependencies
conda create -n iidrs python=3.9
conda activate iidrs
pip install -r requirements.txt
```

## Usage 
Here's a simple example of how to use our code and models:

```
cd recommendationReady
python interact_rl.py
```

## Support 
If you encounter any issues or have questions, please file an issue on this GitHub repository.

## Citation
If you find our dataset, code, or models useful in your research, please consider citing our paper:
```
@inproceedings{10.1145/3581783.3612420,
author = {Zhang, He and Sun, Ying and Guo, Weiyu and Liu, Yafei and Lu, Haonan and Lin, Xiaodong and Xiong, Hui},
title = {Interactive Interior Design Recommendation via Coarse-to-fine Multimodal Reinforcement Learning},
year = {2023},
publisher = {Association for Computing Machinery},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
series = {MM '23}
}
```

