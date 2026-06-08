import csv
import os
import pickle
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="E-Commerce Delivery Predictor", version="1.0.0")

MODEL_PATH = "model.pkl"

WAREHOUSE_VALS  = ['A', 'B', 'C', 'D', 'F']
SHIPMENT_VALS   = ['Flight', 'Road', 'Ship']
IMPORTANCE_VALS = ['high', 'low', 'medium']
GENDER_VALS     = ['F', 'M']

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>E-Commerce Delivery Predictor</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 30px 16px; }
    .container { max-width: 750px; margin: 0 auto; }
    .header { text-align: center; color: white; margin-bottom: 30px; }
    .header h1 { font-size: 2rem; margin-bottom: 8px; }
    .header p { opacity: 0.85; font-size: 1rem; }
    .badge { display: inline-block; background: rgba(255,255,255,0.2); color: white; font-size: 0.75rem; padding: 4px 12px; border-radius: 20px; margin-top: 8px; }
    .card { background: white; border-radius: 20px; padding: 36px; box-shadow: 0 20px 60px rgba(0,0,0,0.15); }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .form-group { display: flex; flex-direction: column; gap: 6px; }
    label { font-size: 0.82rem; font-weight: 600; color: #555; text-transform: uppercase; letter-spacing: 0.5px; }
    input, select { padding: 11px 14px; border: 2px solid #e8e8f0; border-radius: 10px; font-size: 0.95rem; color: #333; transition: border 0.2s; background: #fafafa; }
    input:focus, select:focus { outline: none; border-color: #667eea; background: white; }
    .btn { width: 100%; padding: 15px; background: linear-gradient(135deg, #667eea, #764ba2); color: white; border: none; border-radius: 12px; font-size: 1.05rem; font-weight: 600; cursor: pointer; margin-top: 24px; transition: transform 0.2s, box-shadow 0.2s; }
    .btn:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(102,126,234,0.4); }
    .result { margin-top: 24px; padding: 20px; border-radius: 14px; text-align: center; display: none; animation: fadeIn 0.4s ease; }
    @keyframes fadeIn { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
    .result.success { background: #e8f5e9; border: 2px solid #4caf50; }
    .result.danger  { background: #fdecea; border: 2px solid #f44336; }
    .result .icon  { font-size: 2.5rem; margin-bottom: 8px; }
    .result .label { font-size: 1.3rem; font-weight: 700; color: #333; margin-bottom: 4px; }
    .result .conf  { font-size: 0.9rem; color: #666; }
    .api-link { margin-top: 16px; text-align: center; font-size: 0.85rem; color: #888; }
    .api-link a { color: #667eea; text-decoration: none; font-weight: 600; }
    @media(max-width:560px){ .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>📦 Delivery Predictor</h1>
    <p>E-Commerce Product On-Time Delivery Prediction</p>
    <span class="badge">⚡ Powered by FastAPI</span>
  </div>
  <div class="card">
    <div class="grid">
      <div class="form-group">
        <label>Warehouse Block</label>
        <select id="warehouse_block">
          <option value="A">A</option><option value="B">B</option><option value="C">C</option><option value="D" selected>D</option><option value="F">F</option>
        </select>
      </div>
      <div class="form-group">
        <label>Mode of Shipment</label>
        <select id="mode_of_shipment">
          <option value="Ship" selected>Ship</option><option value="Flight">Flight</option><option value="Road">Road</option>
        </select>
      </div>
      <div class="form-group">
        <label>Customer Care Calls</label>
        <input type="number" id="customer_care_calls" min="0" max="10" value="4" />
      </div>
      <div class="form-group">
        <label>Customer Rating (1-5)</label>
        <input type="number" id="customer_rating" min="1" max="5" value="3" />
      </div>
      <div class="form-group">
        <label>Cost of Product (USD)</label>
        <input type="number" id="cost_of_product" min="0" value="150" />
      </div>
      <div class="form-group">
        <label>Prior Purchases</label>
        <input type="number" id="prior_purchases" min="0" value="3" />
      </div>
      <div class="form-group">
        <label>Product Importance</label>
        <select id="product_importance">
          <option value="low">Low</option><option value="medium" selected>Medium</option><option value="high">High</option>
        </select>
      </div>
      <div class="form-group">
        <label>Gender</label>
        <select id="gender">
          <option value="M" selected>Male</option><option value="F">Female</option>
        </select>
      </div>
      <div class="form-group">
        <label>Discount Offered (%)</label>
        <input type="number" id="discount_offered" min="0" max="100" value="10" />
      </div>
      <div class="form-group">
        <label>Weight (grams)</label>
        <input type="number" id="weight_in_gms" min="0" value="2500" />
      </div>
    </div>
    <button class="btn" onclick="predict()">🔍 Predict Delivery</button>
    <div class="result" id="result">
      <div class="icon"  id="result-icon"></div>
      <div class="label" id="result-label"></div>
      <div class="conf"  id="result-conf"></div>
    </div>
    <div class="api-link">
      📄 <a href="/docs" target="_blank">View API Docs (Swagger UI)</a> &nbsp;|&nbsp;
      <a href="/redoc" target="_blank">ReDoc</a>
    </div>
  </div>
</div>
<script>
async function predict() {
  const payload = {
    warehouse_block:     document.getElementById('warehouse_block').value,
    mode_of_shipment:    document.getElementById('mode_of_shipment').value,
    customer_care_calls: parseInt(document.getElementById('customer_care_calls').value),
    customer_rating:     parseInt(document.getElementById('customer_rating').value),
    cost_of_product:     parseFloat(document.getElementById('cost_of_product').value),
    prior_purchases:     parseInt(document.getElementById('prior_purchases').value),
    product_importance:  document.getElementById('product_importance').value,
    gender:              document.getElementById('gender').value,
    discount_offered:    parseFloat(document.getElementById('discount_offered').value),
    weight_in_gms:       parseFloat(document.getElementById('weight_in_gms').value),
  };
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    const box = document.getElementById('result');
    box.style.display = 'block';
    if (data.error) {
      box.className = 'result danger';
      document.getElementById('result-icon').textContent  = '⚠️';
      document.getElementById('result-label').textContent = 'Error: ' + data.error;
      document.getElementById('result-conf').textContent  = '';
    } else {
      const onTime = data.prediction === 0;
      box.className = 'result ' + (onTime ? 'success' : 'danger');
      document.getElementById('result-icon').textContent  = onTime ? '✅' : '❌';
      document.getElementById('result-label').textContent = data.label;
      document.getElementById('result-conf').textContent  = 'Confidence: ' + data.confidence + '%';
    }
  } catch(e) { alert('Request failed: ' + e.message); }
}
</script>
</body>
</html>"""


class PredictRequest(BaseModel):
    warehouse_block: str
    mode_of_shipment: str
    customer_care_calls: int
    customer_rating: int
    cost_of_product: float
    prior_purchases: int
    product_importance: str
    gender: str
    discount_offered: float
    weight_in_gms: float

    class Config:
        json_schema_extra = {
            "example": {
                "warehouse_block": "B",
                "mode_of_shipment": "Road",
                "customer_care_calls": 6,
                "customer_rating": 5,
                "cost_of_product": 141,
                "prior_purchases": 3,
                "product_importance": "medium",
                "gender": "M",
                "discount_offered": 8,
                "weight_in_gms": 5031
            }
        }


def encode(val, cats):
    return cats.index(val)


def load_and_train():
    X, y = [], []
    with open('E_Commerce.csv', newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            try:
                X.append([
                    encode(row['Warehouse_block'], WAREHOUSE_VALS),
                    encode(row['Mode_of_Shipment'], SHIPMENT_VALS),
                    int(row['Customer_care_calls']),
                    int(row['Customer_rating']),
                    float(row['Cost_of_the_Product']),
                    int(row['Prior_purchases']),
                    encode(row['Product_importance'], IMPORTANCE_VALS),
                    encode(row['Gender'], GENDER_VALS),
                    float(row['Discount_offered']),
                    float(row['Weight_in_gms']),
                ])
                y.append(int(row['Reached.on.Time_Y.N']))
            except Exception:
                continue
    X = np.array(X, dtype=float)
    y = np.array(y)
    split = int(len(X) * 0.8)
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(
        criterion='gini', max_depth=6,
        min_samples_leaf=6, min_samples_split=2,
        random_state=0, class_weight='balanced'
    )
    model.fit(X[:split], y[:split])
    print(f"Model trained. Accuracy: {model.score(X[split:], y[split:]):.4f}")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    return model


if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded from disk.")
else:
    model = load_and_train()


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.post("/predict")
def predict(data: PredictRequest):
    try:
        features = np.array([[
            encode(data.warehouse_block, WAREHOUSE_VALS),
            encode(data.mode_of_shipment, SHIPMENT_VALS),
            data.customer_care_calls,
            data.customer_rating,
            data.cost_of_product,
            data.prior_purchases,
            encode(data.product_importance.lower(), IMPORTANCE_VALS),
            encode(data.gender, GENDER_VALS),
            data.discount_offered,
            data.weight_in_gms,
        ]], dtype=float)

        prediction = int(model.predict(features)[0])
        prob = float(max(model.predict_proba(features)[0]))

        return {
            "prediction": prediction,
            "label": "NOT delivered on time" if prediction == 1 else "Delivered on time",
            "confidence": round(prob * 100, 2)
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
def health():
    return {"status": "ok"}
