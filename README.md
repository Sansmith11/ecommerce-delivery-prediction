# 📦 E-Commerce Product Delivery Prediction

A Machine Learning web app that predicts whether an e-commerce product will be delivered **on time or not**, based on shipment and customer data.

## 🚀 Live Demo
> Deploy on Render and paste your URL here.

---

## 📊 About the Project

- **Dataset**: 10,999 observations, 12 features
- **Target**: `Reached.on.Time_Y.N` — 1 = NOT on time, 0 = On time
- **Best Model**: Decision Tree Classifier (~69% accuracy)
- **Models compared**: Random Forest, Decision Tree, Logistic Regression, KNN, XGBoost

### Key Findings
- Products weighing **2500–3500g** with cost < $250 arrive on time more often
- More customer care calls = higher chance of late delivery
- Discounts > 10% are associated with on-time delivery

---

## 🛠️ Tech Stack
- **Backend**: Flask + Scikit-learn
- **Frontend**: HTML/CSS/JavaScript
- **Deployment**: Render

---

## 📁 Project Structure
```
ecommerce-delivery-prediction/
├── app.py                  # Flask app & ML model
├── E_Commerce.csv          # Dataset
├── ML_Project.ipynb        # Jupyter Notebook (EDA + modeling)
├── templates/
│   └── index.html          # Frontend UI
├── requirements.txt
├── render.yaml             # Render deployment config
└── README.md
```

---

## ⚙️ Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ecommerce-delivery-prediction.git
cd ecommerce-delivery-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```
Open http://localhost:5000 in your browser.

---

## ☁️ Deploy on Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — click **Deploy**
5. Your app will be live at `https://your-app.onrender.com`

---

## 🎓 Input Features

| Feature | Description |
|---|---|
| Warehouse Block | A / B / C / D / F |
| Mode of Shipment | Ship / Flight / Road |
| Customer Care Calls | Number of enquiry calls |
| Customer Rating | 1 (worst) – 5 (best) |
| Cost of Product | In USD |
| Prior Purchases | Number of past orders |
| Product Importance | Low / Medium / High |
| Gender | Male / Female |
| Discount Offered | % discount |
| Weight in grams | Product weight |
