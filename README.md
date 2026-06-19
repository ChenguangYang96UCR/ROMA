# ROMA

ROMA is a multimodal spatio-temporal forecasting framework for urban service
request prediction. The model combines historical time-series signals with
multiple spatial and semantic modalities, including POI embeddings, satellite
or image features, location information, road or spatial adjacency, and
DreamKG/findhelp text-graph features.

The repository also includes scripts for running the full ROMA model, ablation
variants, and external baseline models.

## Repository

```bash
git clone https://github.com/ChenguangYang96UCR/ROMA.git
cd ROMA
```

If SSH is configured:

```bash
git clone git@github.com:ChenguangYang96UCR/ROMA.git
cd ROMA
```

## Project Structure

```text
ROMA/
├── baseline/                 # External baseline repositories and notes
├── data_configs/             # YAML experiment configs
├── data_provider/            # Dataset and dataloader implementation
├── datasets/                 # Data root expected by the loader
├── engines/                  # Training, validation, and testing engines
├── models/                   # ROMA model and layers
├── utils/                    # Metrics, logging, config, and helper utilities
├── run.py                    # Main training/evaluation entry point
├── run_with_svd_vot.sh       # Full ROMA run script
├── run_baseline.sh           # Basic model without ROMA components
├── run_only_temp_baseline.sh # Temporal-only baseline
├── run_without_svd.sh        # Ablation without PMRL-SVD loss
├── run_without_vot.sh        # Ablation without VOT-DPE
├── run_without_confu.sh      # Ablation without ConFu loss
├── run_without_selector.sh   # Ablation without multimodal selector
└── summarize_result.py       # Summarize multi-seed logs
```

## Environment

Create a Python environment and install the required packages:

```bash
conda create -n roma python=3.9
conda activate roma

pip install torch numpy pandas scikit-learn pyyaml tqdm matplotlib transformers thop
```

Some model layers import `flash_attn`. If your environment supports it, install
FlashAttention following the official instructions for your CUDA and PyTorch
versions. If FlashAttention is unavailable, use the branch or local fallback
implementation configured for your environment.

## Data Format

The default data root is configured as:

```yaml
data:
  root_path: datasets
```

For each dataset, ROMA expects files under `datasets/` with the following
layout:

```text
datasets/
├── data/
│   └── <DATASET>/
│       ├── <DATASET>_train1.npz
│       ├── <DATASET>_val1.npz
│       ├── <DATASET>_test1.npz
│       ├── <DATASET>_rn_adj.npy
│       ├── <DATASET>.json
│       └── findhelp_zip_graph.npz
├── poi/
│   └── <DATASET>_poi_vectors.csv
└── picture/
    └── <DATASET>/
        └── image_features.csv
```

The train/validation/test `.npz` files should contain:

- `data`: time-series values with shape `(T, N)`
- `index`: node identifiers used to align POI, image, location, and graph data
- `start_time`: timestamps such as `YYYY-MM-DD HH:MM:SS`

The graph and multimodal files are used as follows:

- `<DATASET>_rn_adj.npy`: spatial or road-network adjacency matrix
- `<DATASET>.json`: node-level latitude and longitude metadata
- `<DATASET>_poi_vectors.csv`: POI embeddings keyed by `list_id`
- `image_features.csv`: satellite/image embeddings keyed by filename
- `findhelp_zip_graph.npz`: DreamKG/findhelp text embeddings and adjacency

Supported Philadelphia service-request datasets used in the experiments include:

```text
philly
Opioid_Response_Unit
Illegal_Dumping
Police_Complaint
Dangerous_Building_Complaint
```

## Configuration

Experiment settings are controlled by YAML files in `data_configs/`.

The main configuration is:

```text
data_configs/roma.yaml
```

To switch datasets, edit the dataset entries:

```yaml
data:
  dataset_map: ["Police_Complaint"]
  train_dataset_list: ["Police_Complaint"]
  val_dataset_list: ["Police_Complaint"]
  test_dataset_list: ["Police_Complaint"]
```

The default ROMA configuration uses:

- input length: `seq_len = 7`
- prediction length: `pred_len = 1`
- batch size: `2`
- encoder dimension: `32`
- decoder dimension: `32`
- number of experts: `4`
- top-k experts: `1`
- learning rate: `0.0004`
- epochs: `20`

## Training

Run the full ROMA model with PMRL-SVD, VOT-DPE, text/image DPE branches,
trend/seasonal decomposition, and ConFu loss:

```bash
bash run_with_svd_vot.sh
```

Useful environment overrides:

```bash
GPU=2 \
CONFIG=data_configs/roma.yaml \
SEEDS="2024 2025 2026" \
bash run_with_svd_vot.sh
```

The full ROMA script enables:

```text
--use_pmrl_svd 1
--lambda_svd 0.1
--svd_tau1 0.05
--svd_tau2 0.05
--use_vot_dpe 1
--use_text_dpe 1
--use_image_dpe 1
--use_trend 1
--use_seasonal 1
--dpe_trend_queries 4
--dpe_seasonal_queries 8
--dpe_num_heads 4
--use_confu_loss 1
--lambda_confu 0.1
```

You can also call `run.py` directly:

```bash
python run.py \
  --gpu 2 \
  --is_training 1 \
  --seed 2024 \
  --config_filename data_configs/roma.yaml \
  --use_pmrl_svd 1 \
  --lambda_svd 0.1 \
  --svd_tau1 0.05 \
  --svd_tau2 0.05 \
  --use_vot_dpe 1 \
  --use_confu_loss 1 \
  --lambda_confu 0.1
```

## Evaluation

To evaluate a saved checkpoint, run:

```bash
python run.py \
  --gpu 2 \
  --is_training 0 \
  --seed 2024 \
  --config_filename data_configs/roma.yaml \
  --eval_model_path checkpoints/<CHECKPOINT_DIR>/model_s2024.pth
```

Training logs are written to `logs/`, and checkpoints/results are written under
the checkpoint path created by the training engine.

## Ablation Studies

The repository provides scripts for common ablation settings:

```bash
bash run_baseline.sh
bash run_only_temp_baseline.sh
bash run_without_svd.sh
bash run_without_vot.sh
bash run_without_confu.sh
bash run_without_selector.sh
```

Each script supports the same environment overrides:

```bash
GPU=2 CONFIG=data_configs/roma.yaml SEEDS="2024 2025 2026" bash run_without_svd.sh
```

## Summarizing Results

After running multiple seeds, summarize the logs with:

```bash
python summarize_result.py \
  --config_filename data_configs/roma.yaml \
  --exp_name svd_vot_tr4_se8 \
  --log_dir logs
```

The summary script reports mean and standard deviation for:

```text
RMSE, MAE, MAPE, SMAPE
```

and writes a CSV file under:

```text
result/
```

For the temporal-only baseline:

```bash
python summarize_only_temp_result.py \
  --config_filename data_configs/only_temp.yaml \
  --log_dir logs
```

## Baselines

External baseline code is documented in:

```text
baseline/baseline_README.md
```

The baseline folder may include:

- BigST
- LargeST
- PatchSTG
- ST-WA

Refer to `baseline/baseline_README.md` for clone commands and reproducibility
notes.

## Reproducibility

Use fixed seeds when running experiments:

```bash
SEEDS="2024 2025 2026" bash run_with_svd_vot.sh
```

The main entry point sets Python, NumPy, and PyTorch seeds through `run.py`.
For paper results, report the mean and standard deviation across all seeds.

## Citation

If you use this repository, please cite the corresponding paper or project once
the citation information is available.
