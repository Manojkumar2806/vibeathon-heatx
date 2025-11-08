# -------------------------------------
# 🔋 Heat-to-Power Prediction using Random Forest (No Auto-Update)
# -------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ------------------------------
# 🔹 1. Train & Save Model
# ------------------------------
def train_model(csv_path="../data/Folds5x2_pp.csv"):
    df = pd.read_csv(csv_path)
    X = df[["AT", "V", "AP", "RH"]]
    y = df["PE"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    joblib.dump(model, "../backend/models/model.pkl")
    print("✅ Model trained and saved as model.pkl")

    return model, {"MAE": mae, "RMSE": rmse, "R2": r2}

# ------------------------------
# 🔹 2. Predict Power Output
# ------------------------------
def predict_power(model, AT, V, AP, RH):
    new_data = pd.DataFrame({"AT": [AT], "V": [V], "AP": [AP], "RH": [RH]})
    predicted_pe = model.predict(new_data)[0]
    print(f"⚡ Predicted Power Output: {predicted_pe:.2f} MW")
    return predicted_pe

# ------------------------------
# 🔹 3. Visualize Model Accuracy
# ------------------------------
def visualize_results(model, csv_path="../data/Folds5x2_pp.csv"):
    df = pd.read_csv(csv_path)
    X = df[["AT", "V", "AP", "RH"]]
    y = df["PE"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    plt.scatter(y_test, y_pred, color="teal", alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--")
    plt.xlabel("Actual Power Output (MW)")
    plt.ylabel("Predicted Power Output (MW)")
    plt.title("Actual vs Predicted Power Output")
    plt.show()

# ------------------------------
# 🚀 Run Main
# ------------------------------
if __name__ == "__main__":
    model, metrics = train_model()
    print(f"🔹 MAE: {metrics['MAE']:.3f}, RMSE: {metrics['RMSE']:.3f}, R²: {metrics['R2']:.3f}")

    # Example prediction
    predict_power(model, AT=28.5, V=62.3, AP=1008.7, RH=48.9)

    # To visualize results (optional)
    # visualize_results(model)
