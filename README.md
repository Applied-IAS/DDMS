# DDMS: Four-Hour Thunderstorm Nowcasting using a Deep Diffusion Model

This repository contains the code for **four-hour thunderstorm nowcasting** using satellite data with a deep diffusion model, as described in our paper. The code supports both **satellite nowcasting** and **convection detection** tasks.

The pre-trained weights for both tasks are publicly available, allowing you to reproduce our results or use the model for your own data.

---

## Table of Contents
- [DDMS: Four-Hour Thunderstorm Nowcasting using a Deep Diffusion Model](#ddms-four-hour-thunderstorm-nowcasting-using-a-deep-diffusion-model)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Environment Setup](#environment-setup)
  - [Data Preparation](#data-preparation)
  - [Usage](#usage)
    - [Testing](#testing)
      - [Satellite nowcasting](#satellite-nowcasting)
      - [Convection detection](#convection-detection)
    - [Training](#training)
    - [Satellite nowcasting](#satellite-nowcasting-1)
  - [Citation](#citation)
  - [License](#license)

---

## Installation

Clone this repository:

```bash
git clone https://github.com/your_username/DDMS.git
cd DDMS
```

## Environment Setup
We recommend using conda to create an isolated environment. A full environment YAML is provided.

```bash
conda env create -f environment.yml
conda activate ddms
```

This will install all necessary packages including:

Python 3.7

PyTorch 1.10.1 with CUDA 11.3

torchvision, torchaudio

ffmpeg, av

numpy, scipy, matplotlib, pandas, scikit-image

and other dependencies listed in environment.yml

⚠️ Note: If av or ffmpeg fail to install via pip, install the system-level dependencies first:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev
```

## Data Preparation

The FengYun satellite data can be download from the website [http://data.nsmc.org.cn/portalsite/default.aspx]. 

Prepare your satellite dataset following the directory structure:

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
```

## Usage
### Testing
Pre-trained weights for satellite nowcasting and convection detection are available in weights/.

Run the testing script:

#### Satellite nowcasting
```bash
python test_video.py --config configs/satellite.yaml --weights weights/satellite_nowcasting.pth
```

#### Convection detection
```bash
python test_convection.py --config configs/convection.yaml --weights weights/convection_detection.pth
```

### Training

### Satellite nowcasting
```bash
python train_video.py --config configs/satellite.yaml
```

Pre-trained Weights
Satellite nowcasting: weights/satellite_nowcasting.pth

Convection detection: weights/convection_detection.pth

You can directly use these weights for testing without retraining.

## Citation
If you use this code for your research, please cite:

```bibtex
@inproceedings{your2025ddms,
  title={Four-Hour Thunderstorm Nowcasting using a Deep Diffusion Model for Satellite Data},
  author={Your Name et al.},
  booktitle={NeurIPS},
  year={2025}
}
```


## License
This repository is released under the MIT License.
