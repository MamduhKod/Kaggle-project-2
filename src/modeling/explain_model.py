import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os

# === Load model and test data ===
model_path = "../../models/best_histgb_model.pkl"
test_data_path = "../../data/interim/x_test.csv"
output_path = "../../reports/figures/SHAP_feature.png"

model = joblib.load(model_path)
df_test = pd.read_csv(test_data_path)


def SHAP_explain(model, df_test):
    # === Compute SHAP values ===
    explainer = shap.Explainer(model)
    shap_values = explainer(df_test)

    # === Generate and save SHAP summary plot ===
    shap.summary_plot(shap_values, df_test, show=False)

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the plot
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"SHAP plot saved to: {output_path}")
