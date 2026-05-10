# Predicting MOOC Course Certification with Classical ML

A small classical machine-learning benchmark on the public **HarvardX + MITx Person-Course De-Identified dataset (HMXPC13)**, exploring how well demographic and engagement features predict whether an enrolled MOOC student earns a course certificate.

**Authors:** Ru Li · Shaobo (Shawn) — collaborative analysis
**Write-up:** Ru Li

The point here is comparison and interpretability rather than chasing state-of-the-art accuracy: we wanted a clean, end-to-end view of the four "default" classical classifiers we'd reach for on a tabular classification problem, plus an honest reflection on the ethics of actually deploying any of them in an edtech setting.

## Models compared

- **Logistic Regression** — baseline linear model, two L2 strengths
- **Decision Tree** — `GridSearchCV` over `max_depth` and `min_samples_leaf`
- **Random Forest** — `GridSearchCV` over `n_estimators`, `max_depth`, `min_samples_leaf`
- **AdaBoost** — three configurations of base-tree depth and learning rate

All models are evaluated on a held-out, stratified 20% validation slice of the labelled data.

## Results

| Model | Train acc | Val acc |
|---|---:|---:|
| Logistic Regression | 0.959 | 0.962 |
| Decision Tree | 0.976 | 0.971 |
| **Random Forest** | **1.000** | **0.974** |
| AdaBoost | 0.979 | 0.974 |

Tree-based ensembles win, but with caveats:

- The Random Forest's training accuracy is 1.0 — a clear overfitting signal — yet validation stays high. The reason is that engagement features (`nevents`, `ndays_act`, `nchapters`, `nplay_video`, `nforum_posts`) are essentially deterministic of certification: "did you finish the course?" is mostly answered by "did you do the work?".
- Logistic Regression is only ~1.2 points worse than the best tree model despite ignoring all feature interactions. For a real product we'd lean toward it for its calibration and interpretability.

## Ethics — short version

The natural commercial application of a model like this is to *target* the demographic slices most likely to certify and concentrate marketing or paid content there. We argue against it in the notebook for three reasons: (1) the model is correlational, not causal — lower completion in a demographic slice usually reflects external constraints (time, money, internet access, language) rather than worth; (2) using model predictions to allocate support creates feedback loops that widen those gaps over time; (3) the dataset is *already* de-identified specifically because the demographic features are sensitive, and any production system should treat them with the same care.

A defensible inversion of the same model: target *additional* support — reminders, mentorship, captioning, deadline flexibility — at students predicted *not* to certify. Same predictions, opposite policy, very different ethical footprint.

## Data

- **Source**: HarvardX + MITx Person-Course Academic Year 2013 De-Identified dataset
- **Citation**: MITx and HarvardX (2014). *HarvardX-MITx Person-Course Academic Year 2013 De-Identified dataset, version 2.0.* Harvard Dataverse, V10. <https://doi.org/10.7910/DVN/26147>

The notebook expects two CSVs (`mooc_train.csv`, `mooc_test.csv`) in the same folder; both are derivable from the Dataverse release with the preprocessing logic in this notebook. They are not committed to this repository because they are large and freely re-downloadable.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
└── mooc_certification_analysis.ipynb
```

## Reproducing

```bash
git clone https://github.com/ruuuuthful/mooc-certification-prediction.git
cd mooc-certification-prediction
pip install -r requirements.txt
# place mooc_train.csv and mooc_test.csv next to the notebook
jupyter lab mooc_certification_analysis.ipynb
```

## License

MIT — feel free to use anything in this repo for your own learning or reference.
