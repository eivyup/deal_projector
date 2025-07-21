# app.py  ───────────────────────────────────────────────────
import streamlit as st, pandas as pd, joblib, pathlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

################################################################################
# 1. Load & clean data (runs once at app start)
################################################################################
HERE = pathlib.Path(__file__).parent
df = pd.read_excel(HERE / "Book1.xlsx")
df = df[df["Deal Status"].isin(["Completed", "Terminated"])].copy()

y = (df["Deal Status"] == "Completed").astype(int)
X = df[["Size", "Deal Premium", "Deal Lenght",
        "Deal Type", "Nature Of Bid", "Payment Type"]]
good = X.notnull().all(axis=1)
X, y = X[good], y[good]

num = ["Size", "Deal Premium", "Deal Lenght"]
cat = ["Deal Type", "Nature Of Bid", "Payment Type"]

pipe = Pipeline([
    ("prep", ColumnTransformer(
        [('ohe', OneHotEncoder(handle_unknown="ignore"), cat)],
        remainder="passthrough")),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
pipe.fit(*train_test_split(X, y, test_size=0.25, random_state=42, stratify=y))

################################################################################
# 2. Streamlit UI
################################################################################
st.title("Deal-Closure Probability")
st.write("Fill in the deal terms and click **Predict**")

with st.form("input"):
    c1, c2 = st.columns(2)

    size          = c1.number_input("Deal Size (USD million)",   value=5000., step=100.)
    premium       = c2.slider       ("Premium (0 – 1 = 0 %–100 %)", 0.0, 1.0, 0.25, 0.01)
    length        = c1.number_input("Expected Length (days)",    value=180,  step=10)

    deal_type     = c2.selectbox("Deal Type",
                                 sorted(df["Deal Type"].dropna().unique()))
    bid_nature    = c1.selectbox("Nature of Bid",
                                 sorted(df["Nature Of Bid"].dropna().unique()))
    payment_type  = c2.selectbox("Payment Type",
                                 sorted(df["Payment Type"].dropna().unique()))

    submitted = st.form_submit_button("Predict")

if submitted:
    sample = pd.DataFrame([{
        "Size": size,
        "Deal Premium": premium,
        "Deal Lenght": length,
        "Deal Type": deal_type,
        "Nature Of Bid": bid_nature,
        "Payment Type": payment_type
    }])
    prob = pipe.predict_proba(sample)[0, 1]
    st.success(f"Estimated probability of **closing**: {prob:.0%}")
