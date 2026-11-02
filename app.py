import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Decision Tree Offer Predictor",
    page_icon="🌳",
    layout="centered",
)

st.title("🌳 Decision Tree Classifier")
st.write("Predict the marketing offer type using your trained Decision Tree model.")

# The app accepts the uploaded model's current filename, or the simpler
# filename decision_tree_model.pkl if you rename it in GitHub.
MODEL_FILES = [
    "decision_tree_model(2).pkl",
    "decision_tree_model(1).pkl",
    "decision_tree_model.pkl",
]

@st.cache_resource
def load_model(model_path: str):
    # Only load pickle files that you created and trust.
    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)

model_path = next((name for name in MODEL_FILES if Path(name).is_file()), None)

if model_path is None:
    st.error(
        "Model file not found. Upload decision_tree_model(2).pkl "
        "(or rename it to decision_tree_model.pkl) into the same GitHub "
        "repository folder as app.py."
    )
    st.stop()

try:
    model = load_model(model_path)
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

st.caption(f"Loaded model: {model_path}")
st.subheader("Enter offer details")

# These are the three features used to train your model.
reward = st.number_input("Reward", min_value=0.0, value=5.0, step=1.0)
difficulty = st.number_input("Difficulty", min_value=0.0, value=10.0, step=1.0)
duration = st.number_input("Duration", min_value=0.0, value=7.0, step=1.0)

if st.button("Predict offer type", type="primary", use_container_width=True):
    input_data = pd.DataFrame(
        [[reward, difficulty, duration]],
        columns=["reward", "difficulty", "duration"],
    )

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted offer type: **{str(prediction).upper()}**")

        if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
            probabilities = model.predict_proba(input_data)[0]
            probability_table = pd.DataFrame(
                {
                    "Offer type": model.classes_,
                    "Probability (%)": (probabilities * 100).round(2),
                }
            ).set_index("Offer type")

            st.subheader("Prediction probabilities")
            st.bar_chart(probability_table)
            st.dataframe(probability_table, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()
st.caption("Model inputs: reward, difficulty, duration.")
