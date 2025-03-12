# 🧠 Brain Tumor Detection using VGG16

A **Streamlit**-based web application that classifies brain tumors from MRI scans using a **VGG16** deep learning model.

## 🚀 Features
- Upload **MRI scans** (PNG, JPG, JPEG) for analysis.
- Uses a **VGG16-based TensorFlow/Keras** model for brain tumor classification.
- Provides **real-time predictions** with confidence scores.
- Interactive UI with **modern styling**.

---

## 🏗️ Project Structure
```
brain-tumor-classification/
│-- tumor_model.keras      # Pre-trained VGG16 model file
│-- app.py                 # Streamlit application
│-- requirements.txt       # Python dependencies
│-- README.md              # Project documentation
```

---

## 🔧 Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ruthvini/brain-tumor-classification.git
cd brain-tumor-classification
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

---

## 📸 Usage
1. **Upload an MRI scan** using the file uploader.
2. The app will preprocess the image and **analyze the scan**.
3. The VGG16 model will classify it as **"Tumor Detected"** or **"No Tumor Detected"**.
4. **Confidence score** will be displayed.

---

## 📜 Dependencies
- `streamlit`
- `tensorflow`
- `numpy`
- `opencv-python`
- `Pillow`

Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## 🎯 Model Details
- The model is a **VGG16-based deep learning model** trained to classify brain tumors.
- It expects **224x224** images as input.
- **Binary Classification**: Tumor vs. No Tumor.


## 👩‍💻 Author
Developed by **Ruthvika Miryala**

Connect with me on [LinkedIn](https://www.linkedin.com/in/ruthvika-miryala) ✨

