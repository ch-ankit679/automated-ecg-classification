# Automated ECG Classification

A Streamlit-based ECG heartbeat classification application powered by a **PyTorch CNN–LSTM with an attention mechanism**. Upload an ECG signal as a CSV file to view the waveform, obtain a five-class prediction, inspect an attention-based heatmap, and compare predictions under simulated signal noise.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" />
  <img alt="PyTorch" src="https://img.shields.io/badge/Deep%20Learning-PyTorch-ee4c2c?logo=pytorch&logoColor=white" />
  <img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-ff4b4b?logo=streamlit&logoColor=white" />
  <img alt="ECG" src="https://img.shields.io/badge/Domain-ECG%20Analysis-7b61a8" />
</p>

> **Important:** This repository is an educational/research prototype. Its predictions and visualizations are not a medical diagnosis and must not be used as a substitute for assessment by a qualified healthcare professional.

## Table of contents

- [Highlights](#highlights)
- [How it works](#how-it-works)
- [Classification labels](#classification-labels)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Input CSV format](#input-csv-format)
- [Run the application](#run-the-application)
- [Deploy with Streamlit Community Cloud](#deploy-with-streamlit-community-cloud)
- [Limitations](#limitations)
- [Dataset reference](#dataset-reference)
- [License](#license)

## Highlights

- **ECG upload and visualization:** Upload a CSV waveform and explore it in an interactive chart.
- **Five-class heartbeat prediction:** Run inference with the saved PyTorch model and view the predicted class and confidence score.
- **CNN–LSTM attention architecture:** Learn local waveform patterns with one-dimensional convolutional blocks and model temporal relationships with an LSTM.
- **Attention-based visualization:** View an attention heatmap overlaid on the signal as an exploratory view of the model's internal weighting.
- **Synthetic noise stress test:** Compare predictions for the original signal, a Gaussian-noise version, and a baseline-wander version.
- **Probability comparison:** Inspect how the predicted class probabilities change across the three signal conditions.

## How it works

The application follows this workflow:

1. **Upload:** The user uploads an ECG signal in CSV format.
2. **Preprocess:** The signal is read as numeric data and converted to a `float32` tensor for inference.
3. **Extract features:** One-dimensional convolutional blocks use batch normalization, Swish activations, a residual connection, and max pooling to learn waveform features.
4. **Model temporal patterns:** An LSTM processes the extracted sequence, and the attention layer combines information from the sequence and recurrent hidden states.
5. **Classify:** A fully connected layer produces probabilities for five heartbeat categories.
6. **Inspect and stress-test:** The UI displays the prediction, an attention-based heatmap, and a comparison of predictions on clean and synthetically perturbed signals.

The attention overlay is a visualization of model-derived weights; it should not be interpreted as a clinically validated explanation of a decision.

## Classification labels

The model returns the following labels, as defined in `predict.py`:

| Label | Description |
|---|---|
| `Normal` | Normal heartbeat class |
| `Atrial Premature` | Atrial premature heartbeat class |
| `PVC` | Premature ventricular contraction class |
| `Fusion (V+N)` | Fusion of ventricular and normal beat class |
| `Fusion (Paced)` | Paced fusion heartbeat class |

These are the labels configured in the application. Their clinical interpretation depends on the dataset, label mapping, and training procedure used for the checkpoint.

## Repository structure

```text
automated-ecg-classification/
├── app.py                       # Streamlit interface, charts, attention plot, noise tests
├── model.py                     # Swish, convolutional blocks, LSTM, attention model, checkpoint loader
├── predict.py                   # Inference and output-label mapping
├── utils/
│   └── preprocessing.py         # CSV preprocessing and synthetic noise functions
├── model/
│   └── ecg_model.pth            # Trained weights expected by the model loader
├── data/                        # Project data assets
├── ecg_app_colab_file.ipynb      # Notebook used during development
└── requirements.txt             # Python dependencies
```

The loader in `model.py` expects the checkpoint at **`model/ecg_model.pth`**, relative to the repository root. If the checkpoint is not present in your local copy or deployment, add the correct trained weights at that path before running the app.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/ch-ankit679/automated-ecg-classification.git
cd automated-ecg-classification
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The application imports Streamlit, PyTorch, NumPy, Pandas, Matplotlib, SciPy, and Altair. If your environment reports that Altair is missing, install it with `pip install altair` and consider adding `altair` as a direct dependency in `requirements.txt`.

## Input CSV format

Upload a **headerless CSV containing numeric ECG samples** in the format expected by the trained model. For example, a single beat can be represented as one row:

```csv
0.12,0.18,0.15,0.08,-0.02,-0.11,-0.05,0.04
```

The values above are illustrative only and are not a clinically meaningful example. Use a complete waveform, not just these few values.

Notes about the current implementation:

- The preprocessing function reads the CSV without a header and uses the **first row** for inference if the file contains multiple rows. For a straightforward test, upload one beat per file.
- Values should be numeric and arranged in the same order and preprocessing scale as the training data.
- The application UI mentions approximately 187 samples, but the model has a fixed LSTM input-size configuration. Ensure that the waveform length is compatible with the checkpoint and model architecture; see [Troubleshooting](#troubleshooting) if you get a tensor-size error.

## Run the application

From the repository root, with the virtual environment activated:

```bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal (usually `http://localhost:8501`). Open it in your browser, upload a compatible ECG CSV file, and review the prediction and visualizations.

## Deploy with Streamlit Community Cloud

1. Push the repository to GitHub, ensuring that `requirements.txt` and the trained checkpoint at `model/ecg_model.pth` are available to the deployment.
2. In [Streamlit Community Cloud](https://streamlit.io/cloud), create an app connected to this repository and branch.
3. Set the **Main file path** to `app.py` and deploy.
4. If you change Python dependencies, update `requirements.txt` and redeploy the app.

Keep the checkpoint path consistent with the relative path used by `model.py`. Do not commit private data, credentials, or other sensitive files to a public repository.

## Limitations

- The current inference preprocessing selects only the first row of a multi-row CSV file.
- Input shape and scale must match the trained model's expectations.
- The noise tests use **synthetically generated** Gaussian noise and baseline wander. They are useful for an exploratory stress test but do not establish clinical-grade noise robustness.
- The probability displayed by the model is not necessarily a calibrated clinical confidence score.
- The application has not been presented here as clinically validated or approved for diagnosis.

## Dataset reference

The project is associated with ECG heartbeat classification and uses a five-class label set. For background on the widely used MIT-BIH Arrhythmia Database, see [PhysioNet — MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/). Check the repository's notebook and data files for the exact dataset version, preprocessing steps, train/test split, and label mapping used to produce the included checkpoint.

## License

No `LICENSE` file is currently visible at the repository root. Add a license if you intend to specify the terms under which others may use, modify, or redistribute this project.
