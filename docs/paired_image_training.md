# Paired image-to-image training guide

This guide walks through the next steps for using `PairedImageDataset` to finetune the diffusion model for MRI→CT style translation (or other paired image tasks) while keeping the autoencoder frozen.

## 1. Arrange the data
```
<data_root>/
  train/
    src/  # MRI projections (source)
    tgt/  # CT projections (target)
  val/
    src/
    tgt/
```
- Filenames must match across `src/` and `tgt/` (e.g., `sample01.png` exists in both).
- Images can be grayscale; the loader converts them to RGB automatically. If your MRI composites are 840×560, keep the CT targets at the same resolution.

## 2. Example dataset instantiation
```python
from ldm.data import PairedImageDataset

train_dataset = PairedImageDataset(
    source_dir="/path/to/data/train/src",
    target_dir="/path/to/data/train/tgt",
    size=(560, 840),           # (height, width); omit to keep originals
    interpolation="bicubic",
    flip_p=0.5,                # optional horizontal augmentation
)
```
The dataset returns a dict with `source`, `target`, and their file paths. Images are normalized to `[-1, 1]` float32 arrays so they can be fed directly into the autoencoder.

## 3. Minimal Hydra config snippet
Create a config (e.g., `configs/latent-diffusion/mri2ct.yaml`) that plugs into `main.py`. Key pieces:
```yaml
model:
  base_learning_rate: 1.0e-4
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    first_stage_config: configs/autoencoder/kl-f4.yaml  # keep this frozen
    first_stage_ckpt_path: models/first_stage_models/kl-f4/model.ckpt
    # ...your diffusion params...

data:
  target: ldm.data.base.DataModuleFromConfig
  params:
    batch_size: 1
    num_workers: 4
    train:
      target: ldm.data.paired_image.PairedImageDataset
      params:
        source_dir: /path/to/data/train/src
        target_dir: /path/to/data/train/tgt
        size: [560, 840]
        interpolation: bicubic
        flip_p: 0.5
    validation:
      target: ldm.data.paired_image.PairedImageDataset
      params:
        source_dir: /path/to/data/val/src
        target_dir: /path/to/data/val/tgt
        size: [560, 840]
        interpolation: bicubic
        flip_p: 0.0
```
Notes:
- `DataModuleFromConfig` will build PyTorch dataloaders for `train` and `validation` using the dataset above.
- Set `size` to your native resolution if you want to avoid resizing; otherwise it resizes with the chosen interpolation.

## 4. Launch training
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --base configs/latent-diffusion/mri2ct.yaml \
  -t --gpus 0,
```
- The autoencoder stays frozen; only diffusion weights update.
- Monitor validation outputs by decoding predicted latents with the autoencoder to ensure CT projections look correct.

## 5. Inference checklist
1. Encode a new MRI composite with the frozen autoencoder to a latent.
2. Run the finetuned sampler (DDIM/PLMS) conditioned on that latent to generate a CT latent.
3. Decode the CT latent with the autoencoder to get the final 2D projection.

Adjust learning rate, batch size, and augmentations as needed based on GPU memory and visual quality.

## 6. Quick "can I run now?" checklist
Before launching training, make sure you have:

- **Data**: The paired directories exist and you can list matching files under both `src/` and `tgt/` (e.g., `ls data/train/src | head`).
- **Config**: A YAML file that points `source_dir`/`target_dir` to those folders and references the correct frozen autoencoder checkpoint.
- **Environment**: Dependencies installed (`conda env create -f environment.yaml && conda activate ldm`) and any required model checkpoints downloaded via `scripts/download_first_stages.sh` and `scripts/download_models.sh`.

If those items are in place, importing `PairedImageDataset` in your config is sufficient—the training script will build the dataloaders automatically via `DataModuleFromConfig` when you run `python main.py ...`.
