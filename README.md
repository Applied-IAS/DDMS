# DDMS: Four-Hour Thunderstorm Nowcasting using a Deep Diffusion Model

![arXiv](https://img.shields.io/badge/arXiv-2404.10512-red)
<!-- ![Project](https://img.shields.io/badge/Project-Page-blue) -->
<!-- ![Blog](https://img.shields.io/badge/Blog-Post-brightgreen) -->
<!-- ![知乎](https://img.shields.io/badge/知乎-详解-orange) -->



This repository contains the code for **four-hour thunderstorm nowcasting** using satellite data with a deep diffusion model, as described in our paper. The code supports both **satellite nowcasting** and **convection detection** tasks.

The pre-trained weights for both tasks are publicly available, allowing you to reproduce our results or use the model for your own data.

> **Note:** This repository is still under active development.

---

## Table of Contents
- [DDMS: Four-Hour Thunderstorm Nowcasting using a Deep Diffusion Model](#ddms-four-hour-thunderstorm-nowcasting-using-a-deep-diffusion-model)
  - [Table of Contents](#table-of-contents)
  - [Abstract](#abstract)
  - [Installation](#installation)
  - [Environment Setup](#environment-setup)
  - [Data Preparation](#data-preparation)
  - [Usage](#usage)
    - [Testing](#testing)
      - [Pre-trained Weights](#pre-trained-weights)
      - [Satellite nowcasting](#satellite-nowcasting)
      - [Convection detection](#convection-detection)
    - [Training](#training)
      - [Satellite nowcasting](#satellite-nowcasting-1)
  - [Citation](#citation)
  - [License](#license)

---

## Abstract
Convection (thunderstorm) develops rapidly within hours and is highly destructive, posing a
significant challenge for nowcasting and resulting in substantial losses to nature and society.
After the emergence of artificial intelligence (AI)-based methods, convection nowcasting has
experienced rapid advancements, with its performance surpassing that of physics-based
numerical weather prediction and other conventional approaches. However, the lead time and
coverage of it still leave much to be desired and hardly meet the needs of disaster emergency
response. Here, we propose deep diffusion models of satellite (DDMS) to establish an AI-based
convection nowcasting system. Specifically, DDMS employs diffusion processes to effectively
simulate complicated spatiotemporal evolution patterns of convective clouds, significantly
improving the forecast lead time. Additionally, it combines geostationary satellite brightness
temperature data and domain knowledge from meteorological experts, thereby achieving
planetary-scale forecast coverage. During long-term tests and objective validation based on the
FengYun-4A satellite, our system achieves, for the first time, effective convection nowcasting up
to 4 hours, with broad coverage (about 20,000,000 km2), remarkable accuracy, and high
resolution (15 minutes; 4 km). Its performance reaches a new height in convection nowcasting
compared to the existing models. In terms of application, our system is highly transferable with
the potential to collaborate with multiple satellites for global convection nowcasting.
Furthermore, our results highlight the remarkable capabilities of diffusion models in convective
clouds forecasting, as well as the significant value of geostationary satellite data when
empowered by AI technologies.

## Installation

Clone this repository:

```bash
git clone git@github.com:Applied-IAS/DDMS.git 
cd DDMS
```

## Environment Setup
We recommend using conda to create an isolated environment. A full environment YAML is provided.

```bash
conda env create -f environment.yml
conda activate ddms
```

This will install all required packages, including:

- Python 3.11

- PyTorch (CUDA 12.4), TorchVision, TorchAudio

- NumPy, SciPy, Matplotlib, Pandas, Scikit-Image

- Xarray, Cartopy, Basemap, Shapely, Proj

- Lightning, Einops, Kornia, Pysteps

- FFmpeg, AV

⚠️ Note: If av or ffmpeg fail to install via pip, install the system-level dependencies first:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev
```

## Data Preparation

The FengYun satellite data can be download from the website [http://data.nsmc.org.cn/portalsite/default.aspx]. 

<!-- Prepare your satellite dataset following the directory structure:

```text
DDMS/
├── data/
│   ├── satellite/
│   │   ├── images/
│   │   └── labels/
│   └── convection/
│       ├── images/
│       └── labels/
Update the data paths in configs/config.yaml if necessary.
``` -->

## Usage
### Testing
#### Pre-trained Weights

- **Satellite nowcasting:**  
  Download from [Google Drive](https://drive.google.com/file/d/1RdxEfT8SJwA_rslraGTA3dETpfGOpp6a/view?usp=sharing), and place it under the path of 
  `./results-mse-10-retrain-nature/resnet-adam-noise-l1-satellite-d64-t1000-residual-alFalse/`

- **Convection detection (local path):**  
  `./gate_unet/dugs-unet-compare-with-rvos-params/best-m-20-0.0008-0.0029-0.9245-all-area.pth.tar`

#### Satellite nowcasting
```bash
python test_video.py --device 0
```

#### Convection detection
```bash
python ./gate_unet/test_written_16_nature_new_color_bar.py --load_path '../results/evaluate/generated/resnet-adam-noise-l1-satellite-d64-t1000-residual-alFalse/pred/'
```

### Training

#### Satellite nowcasting
```bash
python train_video.py 
```

## Citation
If you use this code for your research, please cite:

```bibtex
@article{dai2024four,
  title={Four-hour thunderstorm nowcasting using deep diffusion models of satellite},
  author={Dai, Kuai and Li, Xutao and Fang, Junying and Ye, Yunming and Yu, Demin and Su, Hui and Xian, Di and Qin, Danyu and Wang, Jingsong},
  journal={arXiv preprint arXiv:2404.10512},
  year={2024}
}
```


## License
This repository is released under the [MIT License](./LICENSE). Refer to the License for details.

