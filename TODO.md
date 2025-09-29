# TODO — Recursive Mixture-of-Experts (rMoE) for “Scroll Reading”

Goal: Build a **recursive** MoE that adaptively routes CT-like slices through chains of experts (denoise → ink-seg → unwarp → refine → OCR), stopping when confidence is high. Deliver code, demo, plots, and a short paper.

---

## 0) Project Skeleton

- [ ] Create repo structure
rmoe-scroll/
├── README.md
├── TODO.md
├── env.yml
├── pyproject.toml
├── src/
│ ├── data/
│ │ ├── synth/
│ │ │ ├── generator.py
│ │ │ └── textures/ # papyrus/velum images
│ │ └── datasets.py
│ ├── models/
│ │ ├── experts/
│ │ │ ├── denoise.py
│ │ │ ├── inkseg.py
│ │ │ ├── unwarp.py
│ │ │ ├── refine.py
│ │ │ └── ocr.py
│ │ ├── rmoe.py # recursive routing + fuse
│ │ └── heads.py # confidence, routing, validation
│ ├── train.py
│ ├── eval.py
│ ├── viz.py
│ ├── utils/
│ │ ├── metrics.py
│ │ ├── losses.py
│ │ ├── config.py
│ │ └── logging.py
│ └── api/
│ └── app.py # FastAPI demo
├── scripts/
│ ├── make_data.sh
│ ├── train.sh
│ ├── eval.sh
│ └── demo.sh
├── experiments/
│ ├── configs/
│ │ ├── baseline_flat.yaml
│ │ ├── rmoe_depth1.yaml
│ │ ├── rmoe_depth2.yaml
│ │ └── rmoe_depth3_top2.yaml
│ └── results/ # auto-populated
└── docs/
├── method.md
├── dataset.md
├── experiments.md
└── paper.md


- [ ] `pyproject.toml` with dependencies: `torch`, `torchvision`, `transformers`, `einops`, `opencv-python`, `pillow`, `scikit-image`, `numpy`, `scipy`, `fastapi`, `uvicorn`, `wandb`, `tqdm`, `pyyaml`, `rich`.
- [ ] `env.yml` (conda) or use `uv`/pip; confirm CUDA/Metal.

---

## 1) Synthetic Data Generator

- [ ] Collect historical fonts (Latin/Greek), license-friendly.
- [ ] Place sample corpora (Latin/Greek text) in `data/synth/corpus/`.
- [ ] Implement `generator.py`:
- [ ] Render text onto papyrus/velum textures (vary font, size, ink thickness).
- [ ] Apply geometric warp (simulate scroll curvature; sinusoidal + perspective).
- [ ] Convert to pseudo-CT: blur, banding, phase noise, speckle, fiber overlays.
- [ ] Attenuate ink (low contrast), add occlusions/tears.
- [ ] Output **triplets** per sample:
      - `img`: H×W grayscale slice
      - `mask`: binary ink mask (GT)
      - `text`: ground-truth string
- [ ] CLI: `python -m src.data.synth.generator --n 100000 --out data/synth/out`
- [ ] Train/val/test splits with **held-out corruption types**.

---

## 2) Data Pipeline

- [ ] `datasets.py` returning dicts: `{'img','mask','text','id'}`
- [ ] Augmentations: small rotations, elastic jitter, contrast/noise sweeps.
- [ ] Collate: pad/crop to fixed size; build label sequences for CTC.

---

## 3) Metrics & Losses

- [ ] Metrics:
- [ ] Ink segmentation: IoU/Dice.
- [ ] Text: CER/WER (CTC decoding).
- [ ] Compute: average routing depth, #experts called, FLOPs proxy, latency.
- [ ] Losses:
- [ ] `L_seg = BCE + Dice`
- [ ] `L_text = CTC` (or seq2seq CE if using a small OCR decoder)
- [ ] Routing: load-balancing (aux loss), entropy reg, **validation-aware** penalty.
- [ ] Validation signals:
- [ ] LM perplexity over decoded text (small char-LM).
- [ ] Local 3D consistency proxy (optional: neighborhood smoothness).

---

## 4) Expert Modules (Tiny, Fast)

- [ ] `denoise.py`: UNet-lite (depth 3, ~1–2M params). Input: img; Output: denoised img + conf.
- [ ] `inkseg.py`: UNet-lite segmentation. Input: img (or denoised); Output: mask + conf.
- [ ] `unwarp.py`: small STN or thin-plate spline; Output: rectified img/features + conf.
- [ ] `refine.py`: shallow residual refiner operating on img/mask; Output: refined mask/img + conf.
- [ ] `ocr.py`: tiny CNN/ViT encoder + CTC head; Output: logits, decoded text, conf.
- [ ] Each expert implements interface:
```python
class Expert(nn.Module):
    def forward(self, x, aux=None):
        return {
          'out': tensor_or_text,
          'features': optional_tensor,
          'conf': confidence_scalar,           # 0..1
          'route_logits': optional_tensor      # for sub-routing
        }
