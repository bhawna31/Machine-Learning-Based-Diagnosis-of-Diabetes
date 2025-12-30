# ============================================================
#        MACHINE LEARNING–BASED DIABETES DETECTION APP
#                       Clean & Optimized
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ------------------------------------------------------------
# Streamlit Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="💉",
    layout="wide"
)

# ------------------------------------------------------------
# Load & Preprocess Dataset
# ------------------------------------------------------------
@st.cache_data
def load_dataset():
    """Load and clean the PIMA Diabetes dataset."""
    try:
        df = pd.read_csv("diabetes.csv")
    except FileNotFoundError:
        st.error("❌ ERROR: 'diabetes.csv' not found in the project folder.")
        st.stop()

    # Columns where zero = missing data
    cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    for col in cols_with_zero:
        df[col] = df[col].replace(0, np.nan)
        df[col].fillna(df[col].median(), inplace=True)

    return df


df = load_dataset()

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Standard Scaling (for LR only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------------------
# Train Models (Cached)
# ------------------------------------------------------------
@st.cache_resource
def train_models():
    """Train LR and RF models once and reuse."""
    lr_model = LogisticRegression(
        max_iter=2000, solver="liblinear", random_state=42
    )
    lr_model.fit(X_train_scaled, y_train)

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=7,
        class_weight="balanced",
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    return lr_model, rf_model


lr_model, rf_model = train_models()


# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def predict(model, df, scale=False):
    """Return model prediction label."""
    if scale:
        df = scaler.transform(df)
    pred = model.predict(df)
    return "Diabetic" if pred[0] == 1 else "Non-Diabetic"


def show_metrics(name, model, scaled=False):
    """Show ML model evaluation scores."""
    preds = model.predict(X_test_scaled if scaled else X_test)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    st.write(f"### 📌 {name}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy:.2f}")
    c2.metric("Precision", f"{precision:.2f}")
    c3.metric("Recall", f"{recall:.2f}")
    c4.metric("F1 Score", f"{f1:.2f}")


# ------------------------------------------------------------
# UI — Title & Description
# ------------------------------------------------------------
st.title("💉 Diabetes Prediction System")
st.write("""
This system uses **Machine Learning** algorithms to predict whether a patient is  
**Diabetic** or **Non-Diabetic** based on key medical parameters.

Trained using PIMA Indian Diabetes Dataset with advanced preprocessing.
""")

# ------------------------------------------------------------
# Sidebar — Patient Input Form
# ------------------------------------------------------------
st.sidebar.header("🧪 Enter Patient Health Data")

def get_user_input():
    """Collect user input via sidebar sliders."""
    return pd.DataFrame([{
        "Pregnancies": st.sidebar.slider("Pregnancies", 0, 17, 2),
        "Glucose": st.sidebar.slider("Glucose", 50, 200, 120),
        "BloodPressure": st.sidebar.slider("Blood Pressure", 30, 122, 72),
        "SkinThickness": st.sidebar.slider("Skin Thickness", 0, 99, 20),
        "Insulin": st.sidebar.slider("Insulin", 0, 846, 80),
        "BMI": st.sidebar.slider("BMI", 10.0, 67.1, 31.0),
        "DiabetesPedigreeFunction": st.sidebar.slider("Pedigree", 0.0, 2.42, 0.45),
        "Age": st.sidebar.slider("Age", 18, 100, 29),
    }])


user_data = get_user_input()

# ------------------------------------------------------------
# Prediction Section
# ------------------------------------------------------------
st.subheader("🔍 Prediction")

model_choice = st.selectbox(
    "Choose ML Model",
    ["Random Forest (Recommended)", "Logistic Regression"]
)

if st.button("Predict Diabetes"):
    st.divider()

    if model_choice == "Random Forest (Recommended)":
        result = predict(rf_model, user_data)
    else:
        result = predict(lr_model, user_data, scale=True)

    if result == "Diabetic":
        st.error(f"⚠️ Prediction: **{result}**")
    else:
        st.success(f"✅ Prediction: **{result}**")

    st.write("### Patient Input Summary")
    st.table(user_data)

# ------------------------------------------------------------
# Model Performance Scores
# ------------------------------------------------------------
st.divider()
st.subheader("📊 Model Performance")

show_metrics("Random Forest Classifier", rf_model)
show_metrics("Logistic Regression", lr_model, scaled=True)

# ------------------------------------------------------------
# Conclusion
# ------------------------------------------------------------
st.divider()
st.subheader("📝 Summary")

st.write("""
✔ Random Forest Model gives better performance  
✔ Logistic Regression gives simple and interpretable results  
✔ Proper preprocessing improves accuracy  
✔ Useful for healthcare screening, telemedicine, and diagnosis support  

Future upgrades may include:
- SVM, XGBoost, and deep learning  
- SHAP explainability  
- Real-time API deployment  
- Patient data storage  
""")
