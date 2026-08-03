# 🥔 Potato Leaf Disease Classification

A deep learning web application that detects diseases in potato plant leaves from a single image, using a Convolutional Neural Network (CNN) trained with TensorFlow/Keras and served through a Flask web interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.1-black.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📖 Overview

Potato crops are highly susceptible to diseases such as **Early Blight** and **Late Blight**, which can significantly reduce yield if not detected early. This project provides an end-to-end deep learning solution: a CNN model trained on labeled leaf images, wrapped in a simple web application that lets a user upload a photo of a potato leaf and instantly receive a diagnosis with a confidence score.

## 💡 Motivation

Manual disease diagnosis in agriculture is slow, requires expert knowledge, and isn't always accessible to smallholder farmers. An automated image-based classifier can provide a fast, low-cost first opinion, helping farmers act quickly to protect their crops.

## ❓ Problem Statement

Given an image of a potato leaf, classify it into one of three categories:
- `Early Blight`
- `Late Blight`
- `Healthy`

The model should also be able to flag images that are **not** potato leaves at all, rather than forcing a confident but meaningless prediction.

---

## ✨ Features

- 🧠 Custom CNN trained from scratch (no transfer learning dependency)
- 🌐 Simple, responsive web UI (Flask + Bootstrap 5)
- 📊 Confidence score displayed alongside every prediction
- 🚫 Out-of-distribution guard: if the model isn't confident (`< 80%`), it reports the image as **not a potato leaf** instead of guessing
- ⚡ Fast inference — single image prediction in real time

---

## 🛠️ Technologies Used

| Category | Technology |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Backend | Flask, Werkzeug |
| Numerical Computing | NumPy |
| Frontend | HTML, Bootstrap 5 |
| Training Environment | Jupyter Notebook |

---

## 🏗️ Architecture

The model is a Sequential CNN with the following structure:

```
Input (255x255x3)
   -> Resizing & Rescaling
   -> Data Augmentation (RandomFlip, RandomRotation)   [training only]
   -> Conv2D(32) + MaxPooling2D
   -> Conv2D(64) + MaxPooling2D   (x5 more blocks)
   -> Flatten
   -> Dense(64, activation='relu')
   -> Dense(3, activation='softmax')
```

**Classes:** `Potato___Early_blight`, `Potato___Late_blight`, `Potato___healthy`

---

## 🔄 Workflow

1. Load and label images using `image_dataset_from_directory`
2. Split into train (80%) / validation (10%) / test (10%)
3. Apply data augmentation to the training set only
4. Train the CNN for 20 epochs
5. Save the trained model as `model.keras`
6. Load the model in a Flask app that accepts an uploaded image, preprocesses it, and returns a prediction with a confidence score

---

## 📁 Folder Structure

```
potato-leaf-disease-classification/
│
├── app.py                      # Flask web application
├── model.keras                 # Trained CNN model
├── requirements.txt            # Python dependencies
├── .gitignore
├── LICENSE
├── README.md
│
├── templates/
│   └── index.html              # Web interface
│
├── static/
│   └── .gitkeep                # Stores uploaded images at runtime (ignored by git)
│
└── notebooks/
    └── potato_disease_classification.ipynb   # Model training notebook
```

---

## ⚙️ Requirements

- Python 3.10+
- pip

All Python dependencies are listed in [`requirements.txt`](requirements.txt).

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/nadakhadim33/potato-leaf-disease-classification.git
cd potato-leaf-disease-classification

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## ▶️ How to Run

```bash
python app.py
```

Then open your browser at: `http://127.0.0.1:5000`

## 🖱️ Usage

1. Open the web app in your browser
2. Click **Select potato leaf image** and choose a `.jpg`, `.jpeg`, or `.png` file
3. Click **🔍 Predict Disease**
4. View the predicted class and confidence percentage

## 📤 Example Output

| Input | Prediction | Confidence |
|---|---|---|
| Potato leaf image | `Potato___Early_blight` | 96.4% |

*(See the Screenshots section below for a visual example.)*

---

## 📸 Screenshots

> Add 2–3 screenshots here showing: the upload screen, a healthy-leaf prediction, and a diseased-leaf prediction.
>
> ```markdown
> ![Upload screen](assets/screenshot-upload.png)
> ![Prediction result](assets/screenshot-result.png)
> ```

## 🎬 Demo GIF

> Add a short screen recording (5–10 seconds) showing an upload and prediction in action.
>
> ```markdown
> ![Demo](assets/demo.gif)
> ```

---

## 📊 Model Performance

The model was trained for 20 epochs and achieved:

- **Training Accuracy:** 96.99%
- **Validation Accuracy:** 96.35%
- **Validation Loss:** 0.0564

Training details and evaluation plots are available in the [training notebook](notebooks/potato_disease_classification.ipynb).

## 🗂️ Dataset

The model was trained on the [Potato Disease Dataset](https://www.kaggle.com/datasets/faysalmiah1721758/potato-dataset) from Kaggle, which contains labeled potato leaf images across 3 classes: Early Blight, Late Blight, and Healthy.

After downloading, organize it as follows before running the training notebook:

```
Potato/
├── Potato___Early_blight/
├── Potato___Late_blight/
└── Potato___healthy/
```

> ⚠️ The dataset itself is **not included** in this repository due to its size (2,152 images). Download it from the link above to reproduce training.

---

## 🔮 Future Improvements

- Add unit tests and a CI pipeline (GitHub Actions)
- Deploy the app publicly (e.g., Render, Hugging Face Spaces)
- Add support for more crop types and disease classes
- Replace hardcoded upload storage with unique filenames to avoid collisions
- Add proper MIME-type validation, not just file extension checks
- Add a REST API endpoint (JSON response) alongside the HTML UI

## 🧗 Challenges

- Balancing model confidence thresholds so that non-leaf images are correctly rejected without also rejecting valid, harder-to-classify leaf images
- Ensuring the saved model's preprocessing pipeline exactly matches what the Flask app expects at inference time

## 📚 Lessons Learned

- During training, the model was compiled with `SparseCategoricalCrossentropy(from_logits=True)` while the final layer already applies a `softmax` activation. This mismatch triggered a Keras warning; the correct configuration is `from_logits=False` when the output layer is `softmax`. The model still trained successfully, but this is a good reminder to double check loss/activation pairing when compiling a model.
- Always create upload/storage directories defensively (`os.makedirs(..., exist_ok=True)`) rather than assuming they exist — this was found and fixed in `app.py`.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for improvements, bug fixes, or new features.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 👤 Author

**Nada Khadim**
GitHub: [@nadakhadim33](https://github.com/nadakhadim33)
