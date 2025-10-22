# Humanoid Goalkeeper

[![arXiv](https://img.shields.io/badge/arXiv-2510.18002-brown)](https://arxiv.org/abs/2510.18002)
[![Website 🚀](https://img.shields.io/badge/Website-%F0%9F%9A%80-yellow)](https://humanoid-goalkeeper.github.io/Goalkeeper/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc-sa/4.0/)
[![YouTube 🎬](https://img.shields.io/badge/Youtube-🎬-red)](https://youtu.be/ZK0gPame19M)

Official implementation of the paper **Humanoid Goalkeeper: Learning from Position Conditioned Task-Motion Constraints** by

[Junli Ren](https://renjunli99.github.io/)\*, [Junfeng Long](https://junfeng-long.github.io/)\*, [Tao Huang](https://taohuang13.github.io/), [Huayi Wang](https://why618188.github.io/), [Zirui Wang](https://scholar.google.com/citations?user=Vc3DCUIAAAAJ&hl=zh-TW), [Feiyu Jia](https://trap-1.github.io/), [Wentao Zhang](https://zwt006.github.io/), [Jingbo Wang](https://wangjingbo1219.github.io/)†, [Ping Luo](https://luoping.me/)†, [Jiangmiao Pang](https://oceanpang.github.io/)†

<p align="center">
  <img width="98%" src="docs/teaser.png" alt="Teaser image">
</p>

---

## 🛠️ Installation Instructions

**Clone this repository**
```bash
git clone https://github.com/InternRobotics/Goalkeeper.git
cd Goalkeeper
```
Create a conda environment:
```bash
conda create -n gk python=3.8
conda activate gk
```

Download and install [Isaac Gym](https://developer.nvidia.com/isaac-gym):
```bash
cd isaacgym/python && pip install -e .
```

Install rsl_rl (PPO implementation) and legged gym and other requirements:
```bash
cd rsl_rl && pip install -e . && cd .. 
cd legged_gym &&  pip install -e . && cd .. 
pip install -r requirements.txt
```

## ▶️ Usage
Training:
```bash
cd legged_gym/legged_gym/scripts/
python train.py --exptid=xxx
```

Evaluation:
```bash
cd legged_gym/legged_gym/scripts/
python play.py --exptid=xxx
```

> **Note:** Switch to the `escape` branch to train and evaluate the escape task.

> **Note:** For a quick try, two reference checkpoints are available in `legged_gym/resources/weight/`. 
> Copy one to your experiment log directory (e.g., `logs/<EXPT_ID>/`) to directly evaluate the policy.

## 🧰 Troubleshooting

**ImportError**: `libpython3.8.so.1.0`: cannot open shared object file: No such file or directory
```bash
# Replace /path_to_conda_env_gk with your actual conda env path
sudo cp /path_to_conda_env_gk/lib/libpython3.8.so.1.0 /usr/lib/
```
**CUDA out of memory**
> GPU limitation, try less environments.


## 📈 Training Cost (for reference)

- **GPU:** on RTX 4090 (24G)  
- **Goalkeeper task:** typically ~ **20k episodes** to converge  
- **Escape task:** ~ **40k episodes** for stable jump-escape motion


## ✉️ Contact

For any questions, please email **junlir@connect.hku.hk**. We will respond as soon as possible.



## 📝 Citation

If you find our work useful, please consider citing:
```
@article{ren2025humanoidgoalkeeper,
  title={Humanoid Goalkeeper: Learning from Position Conditioned Task-Motion Constraints},
  author={Ren, Junli, Long, Jungfeng, Huang, Tao and Wang, Huayi, Wang, Zirui and Jia, Feiyu, Zhang, Wentao and Wang, Jingbo, Ping Luo and Pang, Jiangmiao},
  year={2025}
}
```

## 📄 License

The code is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0 International License</a> <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>.
Commercial use is not allowed without explicit authorization.
