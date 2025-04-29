# 📈 Bitcoin Price Prediction Using Machine Learning

This project uses machine learning models—**XGBoost**, **Support Vector Machine (SVM)**, and **Logistic Regression**—to predict Bitcoin price movements. It includes a **Django-based web interface**, complete **documentation**, **presentation**, and **research materials**.

---

## 🔍 Project Overview

The goal of this project is to forecast Bitcoin price trends using historical market data. Machine learning models are trained and evaluated to compare prediction accuracy, and a web interface is provided for real-time user interaction.

---

## 💻 Technologies Used

- **Python 3.7+**
- **Scikit-learn** (SVM, Logistic Regression)
- **XGBoost**
- **Pandas, NumPy, Matplotlib, Seaborn**
- **Django** (Web Interface)
- **Jupyter Notebook / Google Colab**
- **SQLite** (for Django backend)

---

## 📊 Models Implemented

- **XGBoost:** Gradient boosting algorithm used for regression and classification tasks.
- **Support Vector Machine (SVM):** Classification model for non-linear separation.
- **Logistic Regression:** Used as a baseline for binary classification.

---

## 🧪 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- RMSE
- Confusion Matrix

📈 XGBoost achieved the highest accuracy (92%) compared to SVM (85%) and Logistic Regression (81%).

---

## 🌐 Web Interface (Django)

The project includes a user interface built with **Django**, where users can:
- Upload historical Bitcoin data in CSV format.
- View predictions and accuracy.
- Interact with charts and visual outputs.

---

## 📁 Project Structure

```
bitcoin-price-prediction/
├── btc_predictor/          # Django project and app files
├── templates/              # HTML files for the frontend
├── trained_models/         # Saved ML models (optional)
├── notebooks/              # Jupyter notebooks for training/evaluation
├── static/                 # Static files (CSS, JS)
├── media/                  # Uploaded files (if applicable)
├── db.sqlite3              # Django database
├── manage.py               # Django management script
├── requirements.txt        # Python dependencies
├── documentation/          # Project report, PPT, publication, base paper
└── README.md               # Project overview
```

## 📁 Included Files
- 📰 **Base Paper** – Research foundation of the work.
- 📘 **Project Report** – Detailed documentation of methodology and results.
- 📂 **Dataset (CSV Files)** – Historical Bitcoin data used for training and testing machine learning models.
- 📊 **Presentation (PPT)** – Summarizes the project's motivation, models, results, and conclusion.
- 🏅 **Publication** – Research article related to the project.
- 🗂️ **Full Project Code** – Django-based web app with all scripts and notebooks.

---

## 🚀 How to Run

```bash
# Step 1: Clone the repository
git clone https://github.com/DilipVankayala/bitcoin-price-prediction.git
cd bitcoin-price-prediction

# Step 2: Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run Django server
python manage.py runserver

# Access the app at http://127.0.0.1:8000/
```

![Screenshot 2025-04-03 203245](https://github.com/user-attachments/assets/93b3a00d-5e29-43b0-b671-b4da98310506)
![Screenshot 2025-04-03 202703](https://github.com/user-attachments/assets/87708465-1473-4a01-8ad9-35d62c9f3cb0)
![Screenshot 2025-04-03 202828](https://github.com/user-attachments/assets/0cc0cb5d-6388-4731-abbd-2eb8a6f9b393)
![Screenshot 2025-04-03 202731](https://github.com/user-attachments/assets/056e50a6-cd7f-423a-907a-768ccddd116d)

## 📚 Future Scope

- Integrate deep learning models (e.g., LSTM)
- Add social sentiment analysis (Twitter, Reddit)
- Deploy real-time predictions using live APIs
- Extend to other cryptocurrencies like Ethereum

---

## 🤝 Acknowledgements

This project was built as part of an academic/research initiative in financial forecasting using artificial intelligence.

---

## 📬 Contact

For any queries or collaboration:

- **Name:** Dilip Vankayala  
- **Email:** dilipv2003@gmail.com  
- **LinkedIn:** [linkedin.com/in/dilip-vankayala-820a312b2](https://linkedin.com/in/dilip-vankayala-820a312b2)  
- **GitHub:** [github.com/DilipVankayala](https://github.com/DilipVankayala)
                 
