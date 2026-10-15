# Real-Time ASL Fingerspelling Recognition on Consumer Mobile Devices

MSc Computing dissertation project — Atlantic Technological University,
Galway · Faruk Hankanoglu (G00486656) · Supervisor: Dr. Douglas Mota Dias

A fingerspelling recognition system that runs entirely on an ordinary
phone using only its camera: no glove, no depth sensor, no server. Hand
landmarks are extracted with MediaPipe, classified by a small neural
network, and the resulting letter sequence is corrected against a
dictionary using a confidence-weighted noisy-channel model.

**Live demo:**
<https://asl-master-project.vercel.app>

**On Google Play:** [Fingerspell — ASL Reader](https://play.google.com/store/apps/details?id=com.ASLReader)

**Screencast:** [`screencast.mp4`](screencast.mp4) — 7 min 15 s walkthrough of the application in operation

**Dissertation:** [`dissertation.pdf`](dissertation.pdf)

------------------------------------------------------------------------

## Results

| Measurement | Value |
|----|----|
| Letter accuracy, same dataset as training | 99.1 % |
| Letter accuracy, unseen dataset | 71.1 % ± 2.5 |
| Word accuracy, uncorrected | 15.9 % ± 2.4 |
| **Word accuracy, after correction** | **91.7 % ± 2.1** |
| Classifier size | 7,034 parameters (27.5 KB float32) |
| Classification latency, iPhone 12 Pro | 0.14 ms |

Intervals are sample standard deviations over ten retrained seeds. The
correction layer is evaluated on a dataset never used in training, with
the confusion statistics and the evaluation images drawn from disjoint
halves of it.

------------------------------------------------------------------------

## Reproducing the results

The three extracted landmark files are tracked in this repository, so the
experiments run without downloading the image datasets or repeating the
extraction step. Their row counts match Table 3.1 of the dissertation:

| File | Rows | Source dataset |
|---|---:|---|
| `landmarks_train.csv` | 7,348 | Kaggle ASL Alphabet |
| `landmarks_aslhg.csv` | 10,791 | ASL-HG |
| `landmarks_signalphaset.csv` | 7,740 | SignAlphaSet (held out) |

``` bash
pip install -r requirements.txt

# Word-level accuracy of the correction layer, both classifiers
python eval_multiseed_nn_v2.py --model both

# Cross-dataset generalisation experiments
python train_classifier.py --primary landmarks_train.csv --secondary landmarks_signalphaset.csv
python train_with_augmentation.py --kaggle landmarks_train.csv --aslhg landmarks_aslhg.csv --third landmarks_signalphaset.csv
python train_angle_features.py   --kaggle landmarks_train.csv --aslhg landmarks_aslhg.csv --third landmarks_signalphaset.csv
```

`eval_multiseed_nn_v2.py` also accepts
`--holdout {kaggle,aslhg,signalphaset}` for leave-one-dataset-out and
`--mirror` for mirror augmentation.

------------------------------------------------------------------------

## Method

1.  **MediaPipe** locates 21 landmarks on the hand.
2.  Landmarks are **normalised**: the wrist is translated to the origin
    and all coordinates scaled by the wrist-to-middle-fingertip
    distance, removing position and hand size. Orientation is not
    normalised.
3.  A **three-layer network** (63 → 64 → 32 → 26) classifies the letter
    and reports a softmax confidence.
4.  A letter is **committed** after 1.4 s of stable prediction above a
    confidence floor, giving roughly fourteen consecutive predictions
    per committed letter at the configured inference rate. These are
    successive frames of a held pose, not independent samples: they are
    correlated, so the window buys stability against momentary dropouts
    rather than fourteen times the evidence.
5.  The word is **corrected** against a 7,228-word dictionary. Each
    candidate substitution costs `(1 − f[c][p]) · conf(p)`, where
    `f[c][p]` is the measured probability that true letter `c` is
    predicted as `p`. Confusions the classifier actually makes are cheap
    to undo; letters it was confident about resist change.

The confusion matrix is estimated on one half of the held-out dataset
and the correction evaluated on the other, disjoint half, so the
correction layer is never scored on the errors it was calibrated from.

------------------------------------------------------------------------

## Repository contents

### Pipeline

| File | Purpose |
|------------------------------------|------------------------------------|
| `extract_landmarks.py` | Images → 21 MediaPipe hand landmarks → CSV |
| `crop_hands.py` | Images → cropped hand regions, for the CNN baseline |
| `angle_features.py` | Landmarks → 19 rotation-invariant joint angles |

### Training and experiments

| File                         | Purpose                                      |
|------------------------------------|------------------------------------|
| `train_classifier.py`        | Train on one dataset, evaluate cross-dataset |
| `train_combined.py`          | Train on two datasets                        |
| `train_with_augmentation.py` | Mirror augmentation                          |
| `train_angle_features.py`    | Joint-angle representation                   |
| `train_cnn.py`               | MobileNetV3-Small on cropped images          |
| `test_third_dataset.py`      | Evaluate a saved model on a held-out dataset |

### Correction layer and evaluation

| File | Purpose |
|------------------------------------|------------------------------------|
| `eval_multiseed_nn_v2.py` | Main evaluation: multi-seed word accuracy, both classifiers, circularity-safe split |
| `evaluate_word_level_accuracy_final.py` | Single-run word-level evaluation |
| `evaluate_word_level_accuracy_multiseed.py` | Earlier multi-seed version (Random Forest) |
| `word_builder_demo.py` | Command-line demonstration of the word builder |

### Deployment

| File | Purpose |
|------------------------------------|------------------------------------|
| `export_model_for_web.py` | Keras model → `model_weights.json` for the browser |
| `model_weights.json` | Trained network weights, 7,034 parameters |
| `app/` | The deployed progressive web application |

### Data

| File | Purpose |
|------------------------------------|------------------------------------|
| `landmarks_train.csv` | Landmarks from the Kaggle ASL Alphabet dataset |
| `landmarks_aslhg.csv` | Landmarks from ASL-HG |
| `landmarks_signalphaset.csv` | Landmarks from SignAlphaSet, held out |
| `words10k.txt` | Dictionary used by the correction layer |

### Submission

| File | Purpose |
|------------------------------------|------------------------------------|
| `dissertation.pdf` | The dissertation as submitted, 58 pages |
| `screencast.mp4` | Screencast of the application in operation, 7 min 15 s |
| `requirements.txt` | Python dependencies for the scripts above |

See `README_run_locally.md` for environment setup and for regenerating
the landmark CSVs from the source images.

------------------------------------------------------------------------

## Source datasets

The image datasets are not redistributed here; only the extracted
landmark coordinates are included.

| Dataset | Role | Source |
|---|---|---|
| Kaggle ASL Alphabet | Training | <https://www.kaggle.com/datasets/grassknoted/asl-alphabet> |
| ASL-HG | Training | <https://doi.org/10.17632/j4y5w2c8w9.1> |
| SignAlphaSet | Held out | <https://doi.org/10.17632/8fmvr9m98w.3> |

------------------------------------------------------------------------

## Version

Results reported in the dissertation correspond to tag
`v2.1-dissertation`, which is the first tag to hold the landmark CSVs,
`requirements.txt`, the dissertation PDF and the screencast together with
the code. Neither earlier tag covers the numbers in the submitted
document: `v1.0-results` predates `eval_multiseed_nn_v2.py`, which
produced the word-level and augmentation figures, and
`v2.0-dissertation` predates the landmark CSVs and the PDF.
