
from pathlib import Path
import joblib

model_path = Path("xgboost_fraud_model.pkl")

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

model = joblib.load(model_path)

print("MODEL TYPE:", type(model).__name__)
print("FEATURE NAMES:", getattr(model, "feature_names_in_", None))
print("NUMBER OF FEATURES:", getattr(model, "n_features_in_", None))
print("CLASSES:", getattr(model, "classes_", None))

if hasattr(model, "get_booster"):
    booster = model.get_booster()
    print("BOOSTER FEATURES:", booster.feature_names)



import pandas as pd

sample = pd.DataFrame([{
    "step": 100,
    "amount": 185000.0,
    "oldbalanceOrg": 185000.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
    "errorBalanceOrig": 0.0,
    "errorBalanceDest": 185000.0,
    "type_TRANSFER": 1
}])

probability = model.predict_proba(sample)[0, 1]
prediction = model.predict(sample)[0]

print("\nTEST PREDICTION")
print(f"Fraud probability: {probability:.4%}")
print(f"Predicted class: {prediction}")

