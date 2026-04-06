import io
import openpyxl
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import scipy
import sklearn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import xgboost


def filter_cols(df):
  df = df.copy()
  df = df[df['MartialStatus'] != 'Unknown'].reset_index(drop=True)
  df = df[df['Gender'] != 'XNA'].reset_index(drop=True)
  df['Occupation'] = df['Occupation'].replace(['Unemployed', 'Student', 'Businessman', 'Maternity leave'], 'Other')
  df['RiskLevel'] = df['RiskLevel'].fillna('Unknown')
  print(f"Duplicates found: {df.duplicated().sum().sum()}")
  df = df.drop_duplicates()
  # Separate features if fraud column is present
  if 'Fraud' in df.columns:
    X, y = df.drop(columns=['Fraud'], axis=1), df['Fraud']
  else:
    X = df
    y = None
  return X, y


def map_cols(df):
  df = df.copy()
  risk_order = {'LowRisk': 0, 'Risk': 1, 'HighRisk': 2, 'Unknown': -1}
  edu_order = {'Lower secondary': 0, 'Secondary / secondary special': 1, 'Incomplete higher': 2, 'Higher education': 3, 'Academic degree': 4}
  gender_map = {'M': 0, 'F': 1}
  df['RiskLevel'] = df['RiskLevel'].map(risk_order) # Ordinal Encoder
  df['Education'] = df['Education'].map(edu_order) # Ordinal Encoder
  df['Gender']= df['Gender'].map(gender_map)
  return df


def transform_num_cols(df):
  df = df.copy()
  drop_cols = ['LoanID', 'NumberOfBankAccounts', 'ExperianRating']
  df = df.drop(drop_cols, axis = 1)
  return df

## LOAD MODEL
@st.cache_resource
def load_model():
  # Ensure your model file is in the same directory
  base_path = os.path.dirname(__file__)
  model_path = os.path.join(base_path, 'fraud_classificator_joblib.joblib')
  return joblib.load(model_path)

model = load_model()

## STREAMLIT UI
st.title("🛡️ Fraud Detection System")
st.markdown("Upload a CSV file containing applicant data to identify potential fraud risks.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read data
    input_df = pd.read_csv(uploaded_file)
    X, y = filter_cols(input_df)

    upload_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("fraud_app_upload_key") != upload_key:
        st.session_state["fraud_app_upload_key"] = upload_key
        st.session_state["fraud_app_results"] = None

    st.write("### Data Preview")
    st.dataframe(X.head())
    if y is not None:
        st.write("### Target Variable Preview")
        st.dataframe(y.head())

    if st.button("Run Fraud Analysis"):
        try:
            data_to_predict = X.copy()
            predictions = model.predict(data_to_predict)
            probabilities = model.predict_proba(data_to_predict)[:, 1]
            results_df = X.copy()
            results_df["Is_Fraud"] = [
                "Human Review" if p >= 0.5 and p <= 0.78 else "Fraud" if p > 0.78 else "Not Fraud"
                for p in probabilities
            ]
            results_df["Fraud_Probability"] = [f"{p:.2%}" for p in probabilities]
            st.session_state["fraud_app_results"] = {
                "results_df": results_df,
                "input_columns": list(input_df.columns),
                "y": y,
                "predictions": predictions,
                "probabilities": probabilities,
            }
        except Exception as e:
            st.session_state["fraud_app_results"] = None
            st.error(f"An error occurred during processing: {e}")
            st.info("Ensure the CSV has all the required columns used during model training.")

    cached = st.session_state.get("fraud_app_results")
    if cached is not None:
        results_df = cached["results_df"]
        input_columns = cached["input_columns"]
        y_cached = cached["y"]
        predictions = cached["predictions"]

        if y_cached is not None:
            cm = confusion_matrix(y_cached, predictions, labels=[0, 1])
            sns.set_theme(style="whitegrid")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                xticklabels=["Not Fraud", "Fraud"],
                yticklabels=["Not Fraud", "Fraud"],
                annot_kws={"size": 14},
            )
            ax.set_title("Confusion Matrix: Fraud Detection Analysis", fontsize=12)
            ax.set_xlabel("Predicted Labels", fontsize=10)
            ax.set_ylabel("True Labels", fontsize=10)
            st.write("### Confusion Matrix")
            st.pyplot(fig)
            st.write("### Classification Report")
            report = classification_report(y_cached, predictions)
            st.code(report, language="text")
            st.write("### Feature Importance")
            feature_importances = model.named_steps["model"].feature_importances_
            feature_names = model.named_steps["model"].feature_names_in_
            feature_importances_df = pd.DataFrame(
                {"Feature": feature_names, "Importance": feature_importances}
            ).sort_values(by="Importance", ascending=False)
            top_10_features = feature_importances_df.head(10)
            sns.set_theme(style="whitegrid")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=top_10_features,
                x="Importance",
                y="Feature",
                ax=ax,
                palette="viridis",
            )
            ax.set_title("Top 10 Most Influential Features for Fraud Detection", fontsize=16)
            ax.set_xlabel("Importance Score", fontsize=14)
            ax.set_ylabel("Feature", fontsize=14)
            sns.despine(left=True, bottom=True)
            st.pyplot(fig)

        st.success("Analysis Complete!")
        st.write("### Prediction Results")
        vc = results_df["Is_Fraud"].value_counts(normalize=True)
        st.write(
            f"Not Fraud: {results_df['Is_Fraud'].value_counts()['Not Fraud']} ({vc['Not Fraud']:.2%})"
        )
        st.write(f"Fraud: {results_df['Is_Fraud'].value_counts()['Fraud']} ({vc['Fraud']:.2%})")
        st.write(
            f"Human Review: {results_df['Is_Fraud'].value_counts()['Human Review']} ({vc['Human Review']:.2%})"
        )
        st.dataframe(
            results_df[["Is_Fraud", "Fraud_Probability"] + [c for c in input_columns if c != "Fraud"]]
        )

        download_option = st.selectbox(
            "Download Options", ["All Predictions", "Human Review"], key="fraud_download_option"
        )
        format_option = st.selectbox(
            "File format", ["CSV", "Excel"], key="fraud_download_format"
        )

        if download_option == "All Predictions":
            out_df = results_df
            base_name = "final_fraud_predictions"
        else:
            out_df = results_df[results_df["Is_Fraud"] == "Human Review"]
            base_name = "human_review_predictions"

        if format_option == "CSV":
            file_data = out_df.to_csv(index=False).encode("utf-8")
            file_name = f"{base_name}.csv"
            mime = "text/csv"
        else:
            buf = io.BytesIO()
            out_df.to_excel(buf, index=False, engine="openpyxl")
            file_data = buf.getvalue()
            file_name = f"{base_name}.xlsx"
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        dl_key = f"fraud_dl_{'all' if download_option == 'All Predictions' else 'human'}_{format_option.lower()}"
        st.download_button(
            label=f"Download {download_option} as {format_option}",
            data=file_data,
            file_name=file_name,
            mime=mime,
            key=dl_key,
        )


