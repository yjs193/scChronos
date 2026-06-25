# scChronos: Transformer-based modeling of temporal gene expression dynamics in single-cell transcriptomes

**scChronos** is a reference-conditioned Transformer framework for temporal single-cell transcriptome recovery and forecasting. 

------

## 🌟 Key Innovations

- **Reference-Conditioned Temporal Modeling**
- **Target-Specific Reference Attention**
- **Local-Global Fusion Decoder** 

------

## 📁 Repository Structure

```
scChronos/
├── configs/
├── scchronos/
│   ├── __init__.py
│   ├── data.py
│   ├── evaluate.py
│   ├── foundation.py
│   ├── losses.py
│   ├── model.py
│   ├── train_utils.py
│   └── vocab.py
├── scripts/
│   ├── train_scchronos.py
│   └── predict_scchronos.py
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

------

## 📦 Dataset Descriptions

| **Dataset**                    | **Species**            | **Tasks**                                                    | **Description** |
| -------------------------------| ---------------------- | ------------------------------------------------------------ |  ------------------------------------------------------------ | 
| **Schiebinger2019**            | Mouse                  | Recovery / Forecasting                                    | Time-resolved single-cell reprogramming dataset |
| **ECED_Kidney** | Mouse        | Recovery / Forecasting | Embryonic kidney developmental time-series dataset |
| **Veres**                      | Human                  | Recovery / Forecasting                          | Human differentiation-related single-cell time-series dataset |
| **Cao2020**                    | Human                  | Recovery / Forecasting                           | Human fetal developmental single-cell time-series dataset |


## 🚀 Quick Start
### Installation
```
git clone git@github.com:yjs193/scChronos.git
cd scChronos
conda create -n scchronos python=3.11 -y
conda activate scchronos
pip install -r requirements.txt
```

### Training

```
CUDA_VISIBLE_DEVICES=0 python scripts/train_scchronos.py \
  --config configs/eced_kidney_recovery_hvg1000.yaml \
  --wandb
```


### Prediction

```
CUDA_VISIBLE_DEVICES=0 python scripts/predict_scchronos.py \
  --config configs/schiebinger2019_recovery_hvg1000.yaml \
  --checkpoint runs/schiebinger2019_recovery_hvg1000/xxx.pt \
```

### 📧 Contact

For questions, please contact J. Yao (csyjs@mail.scut.edu.cn)
