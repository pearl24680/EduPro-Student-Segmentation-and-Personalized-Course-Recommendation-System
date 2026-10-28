# EduPro – Student Segmentation & Course Recommendation (Streamlit)

## Run locally
1. Install Python 3.10+
2. In this folder:  `pip install -r requirements.txt`
3. `streamlit run app.py`   (browser opens at http://localhost:8501)

Keep `EduPro_Online_Platform.xlsx` in the same folder as `app.py`
(or upload it from the sidebar).

## Files
- app.py – Streamlit dashboard (4 tabs: Learner Explorer, Cluster Dashboard, Segment Comparison, About)
- core.py – segmentation (K-Means, K=5) and recommender
- features.py – learner-level feature engineering
- pipeline.py – offline analysis: metrics, figures, CSVs (`python pipeline.py`)

## Deploy free (Streamlit Community Cloud)
1. Upload this folder to a GitHub repo (include the .xlsx and requirements.txt)
2. Go to share.streamlit.io -> New app -> select repo -> main file: app.py -> Deploy
