# Ubuntu (WSL) Setup for Mamba Training

## 1) Install WSL + Ubuntu

```bash
wsl --install -d Ubuntu
```

## 2) Create Python 3.10 venv

```bash
cd ~
rm -rf mamba_env
python3.10 -m venv /mnt/d/mamba_env_ha
source /mnt/d/mamba_env_ha/bin/activate

```

## 3) Open project in WSL

```bash
cd /mnt/d/KLTN/KLTN_Mamba
```

## 4) Install PyTorch (CUDA)

```bash
TMPDIR=/mnt/d/tmp pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 5) Install CUDA Toolkit (WSL)

```bash
sudo apt update
sudo apt install -y nvidia-cuda-toolkit
```

## 6) Install Mamba dependencies

```bash
pip install causal-conv1d>=1.4.0 --no-build-isolation
pip install mamba-ssm --no-build-isolation
pip install --upgrade torchvision torchaudio
```

## 7) Install project requirements

```bash
pip install -r requirements.txt
```

## 8) Train (example)

```bash
python src/mamba/train_mamba_aqi.py \
  --data-path dataset/air_quality.csv \
  --epochs 10 \
  --window-size 72 \
  --horizon 12 \
  --batch-size 128 \
  --device cuda \
  --amp
```

## Notes

- If you hit PyTorch conflicts, remove old torch packages and reinstall the CUDA wheel.
- If you want to open with VS Code WSL, use the "Connect to WSL" option and open:
  /mnt/d/KLTN/KLTN_Mamba
