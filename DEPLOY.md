# WaferSense AI: Full Upload and Deployment Guide (GitHub + Streamlit Community Cloud)

## 0. What is in this folder

```
WaferSense/
├── app.py                 # Streamlit host
├── site_template.html     # the whole website
├── export_web.py          # builds artifacts/webdata.json
├── common.py, train.py, WaferSense_Colab.ipynb
├── requirements.txt       # deployment (streamlit only)
├── requirements-dev.txt   # local/Colab (torch etc.)
├── .streamlit/config.toml
├── README.md              # results already filled in
└── artifacts/
    ├── model.pt
    ├── metrics.json       # your tested results
    └── samples.npz
```

## 1. Test on your computer (one time)

```bash
cd WaferSense
python -m venv venv
venv\Scripts\activate            # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements-dev.txt
python export_web.py             # creates artifacts/webdata.json
streamlit run app.py             # opens http://localhost:8501
```

Check that the Model Performance section shows Accuracy 92.05%, Macro-F1 90.10%, Balanced accuracy 92.29%.

`webdata.json` is the key step: it lets the deployed app run without torch.

## 2. Upload to GitHub

1. Create an account at github.com, then click **New repository**. Name it `wafersense-ai`, set it **Public**, and do not tick "Add README".
2. Install Git (git-scm.com) and run inside the WaferSense folder:

```bash
git init
git add .
git commit -m "WaferSense AI: CNN wafer map classifier with Grad-CAM"
git branch -M main
git remote add origin https://github.com/<your-username>/wafersense-ai.git
git push -u origin main
```

3. Refresh the repo page. Make sure `artifacts/webdata.json` is there (about 3 MB, well under GitHub's 100 MB limit).

No Git? Use **Add file → Upload files** on GitHub and drag in everything, including the `artifacts` and `.streamlit` folders. Hidden folders can be skipped by the drag-drop, so check `.streamlit/config.toml` made it (the app still works without it, only the theme changes).

## 3. Deploy on Streamlit Community Cloud

1. Go to share.streamlit.io and sign in with GitHub.
2. Click **Create app → Deploy a public app from GitHub**.
3. Repository: `<your-username>/wafersense-ai`, Branch: `main`, Main file path: `app.py`.
4. Optional: choose a custom subdomain (e.g. `wafersense-ai`), giving `https://wafersense-ai.streamlit.app`.
5. Click **Deploy**. The first build takes 1 to 3 minutes.

Then update the "Live demo" link in README.md with your real URL and push again.

## 4. Updating later

Change files, then `git add . && git commit -m "update" && git push`. Streamlit redeploys automatically. If you retrain the model, run `python export_web.py` again before committing.

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| "artifacts/webdata.json is missing" | Run `python export_web.py` locally and push the file |
| App shows empty live-model section | Same as above (webdata.json not found) |
| `ModuleNotFoundError: torch` on the cloud | You are using requirements-dev.txt as requirements.txt; use the streamlit-only one |
| App went to sleep | Free apps sleep after inactivity; click "Yes, get this app back up" |
| Push rejected (file too large) | Files over 100 MB are blocked; none here come close |
