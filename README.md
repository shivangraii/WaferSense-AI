# 🔬 WaferSense AI

**Explainable semiconductor wafer-map defect classification** — a software-only prototype that classifies wafer maps into defect-pattern categories and shows *where* the model looked using Grad-CAM heatmaps.

🌐 **Live demo:** https://wafersense.streamlit.app

> Decision-support prototype only. It recognises spatial patterns on labelled wafer maps; it does not inspect physical wafers, diagnose root causes, or certify quality.

---

## Features

| Page | What it does |
|---|---|
| **Home** | Project overview, dataset summary, responsible-use note |
| **Wafer Inspector** | Pick a held-out demo wafer or upload your own map → predicted class, confidence, class probabilities, low-confidence warning |
| **Explainability** | Grad-CAM heatmap and overlay for the predicted (or any selected) class |
| **Analytics** | Class distribution, train/val/test split sizes, session prediction log |
| **Model Performance** | Accuracy, macro-F1, balanced accuracy, per-class precision/recall/F1, confusion matrix, training curves, CNN vs Random Forest baseline |
| **Report Export** | Downloadable PDF inspection summary |

## Dataset

**WM-811K** (Wu et al., *IEEE Transactions on Semiconductor Manufacturing*, 2015) — 811,457 wafer maps, of which 172,950 carry a failure-type label.

- Classes (9): Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none
- Encoding: 0 = background, 1 = pass die, 2 = fail die
- Source: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map (check the dataset licence before reuse)
- The raw dataset (~2 GB) is **not** included in this repo.

## Method

1. **Filter** to labelled maps only (unlabelled records are dropped).
2. **Subset:** cap each class at 6,000 samples (mainly shrinks the dominant `none` class) → 27,839 maps used.
3. **Preprocess:** nearest-neighbour resize to 64×64 (keeps categorical values intact), one-hot encode into 3 channels.
4. **Split:** stratified 70 / 15 / 15 (train / validation / test). The test set is evaluated once, at the end.
5. **Baseline:** Random Forest on pooled die-density features.
6. **Model:** 4-block CNN (Conv-BN-ReLU), global average pooling, linear head.
7. **Imbalance handling:** class-weighted cross-entropy; augmentation limited to label-preserving rotations/flips.
8. **Training:** AdamW + OneCycle learning rate, early stopping on validation macro-F1.
9. **Explainability:** Grad-CAM on the last convolutional block.

## Results

Evaluated once on the held-out test set (the app's **Model Performance** section shows the full confusion matrix and training curves).

| Model | Accuracy | Macro-F1 | Balanced accuracy |
|---|---|---|---|
| **CNN** | **92.05%** | **90.10%** | **92.29%** |
| Random Forest (baseline) | 80.34% | 75.47% | 73.17% |

Test set: 4,176 wafer maps (15% stratified hold-out of a 27,839-map subset), 40 epochs trained. The CNN beats the baseline by about 11.7 points of accuracy and 14.6 points of macro-F1.

**Per-class results (CNN, test set)**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Center | 0.950 | 0.943 | 0.946 | 644 |
| Donut | 0.806 | 0.904 | 0.852 | 83 |
| Edge-Loc | 0.916 | 0.879 | 0.897 | 779 |
| Edge-Ring | 0.986 | 0.982 | 0.984 | 900 |
| Loc | 0.850 | 0.822 | 0.836 | 539 |
| Near-full | 0.846 | 1.000 | 0.917 | 22 |
| Random | 0.855 | 0.954 | 0.902 | 130 |
| Scratch | 0.802 | 0.883 | 0.840 | 179 |
| none | 0.930 | 0.940 | 0.935 | 900 |

Strongest: Edge-Ring (F1 0.984) and Center (0.946). Weakest: Loc (0.836), Scratch (0.840) and Donut (0.852). Loc is most often confused with Edge-Loc, Center and Scratch (see confusion matrix in the app). Near-full and Donut have very few test samples (22 and 83), so their scores are noisy.

Macro-F1 and balanced accuracy are reported alongside accuracy because the classes are heavily imbalanced.

## Repository structure

```
├── app.py                  # Hosts the one-page site (Streamlit is only the host)
├── site_template.html      # The whole website: landing, live model, explainability, analytics, performance, report
├── export_web.py           # Turns artifacts/ into artifacts/webdata.json for the in-browser model
├── common.py, train.py     # Model, Grad-CAM reference, training pipeline
├── WaferSense_Colab.ipynb  # Train on a free Colab GPU
├── .streamlit/config.toml  # Dark theme
├── requirements.txt        # deployment (streamlit only)
├── requirements-dev.txt    # local / Colab (torch etc.)
├── DEPLOY.md               # step-by-step GitHub + Streamlit Cloud guide
└── artifacts/              # model.pt, metrics.json, samples.npz (and optionally webdata.json)
```

The trained CNN runs inside the page (BatchNorm folded into the convolutions, Grad-CAM computed as
ReLU(sum of class weights x last feature maps), checked against PyTorch to ~1e-7).

## Run locally

```bash
pip install -r requirements-dev.txt
python export_web.py      # once: creates artifacts/webdata.json
streamlit run app.py
```

The app needs the `artifacts/` folder (included in this repo).

## Retrain the model

**Colab (recommended):** open `WaferSense_Colab.ipynb`, enable a GPU runtime, upload `common.py` and `train.py`, and run the cells. Download the generated `artifacts.zip` and replace the `artifacts/` folder.

**Locally:**
```bash
python train.py --data /path/to/folder_with_LSWMD.pkl --epochs 40 --max_per_class 6000
```

## Uploading your own wafer map

In **Wafer Inspector → Upload wafer map**:
- `.npy` or `.csv`: a 2D array with values 0 (background), 1 (pass die), 2 (fail die)
- Image (`.png` / `.jpg`): converted approximately — dark → background, mid-tone → pass, bright → fail

## Limitations

- Trained and evaluated on a single public dataset; results may not generalise to other fabs, tools or process nodes.
- The train/val/test split is stratified by label, not grouped by lot, so lot-level leakage is possible.
- Maps are resized to 64×64, which can lose fine detail (e.g. thin scratches).
- Grad-CAM is a coarse inspection aid, not proof of causal reasoning or of prediction correctness.
- Not validated for production control or quality certification.

## Tech stack

Python · PyTorch · scikit-learn · Streamlit · NumPy · pandas · Matplotlib

## Acknowledgements

- WM-811K dataset: Wu, M.-J., Jang, J.-S. R., & Chen, J.-L. (2015). *Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets.* IEEE Trans. Semiconductor Manufacturing, 28(1), 1–12.
- Built as a capstone project for the Samsung Innovation Campus AI course.
