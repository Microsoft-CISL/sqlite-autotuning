import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from mlos_bench.storage import from_config


# Load the storage configuration
@st.cache_resource
def load_storage():
    return from_config(config="storage/sqlite.jsonc")


storage = load_storage()

# Sidebar for Experiment Selection
st.sidebar.title("Azure MySQL Config Analyzer")
experiment_id = st.sidebar.selectbox("Select Experiment", options=storage.experiments.keys())

# Load selected experiment
exp = storage.experiments[experiment_id]
df = exp.results_df

st.title(f"Azure MySQL Experiment: {experiment_id}")
st.write(f"Description: {exp.description}")

# Metrics and Columns
config_columns = [col for col in df.columns if col.startswith("config.")]
result_columns = [col for col in df.columns if col.startswith("result.")]
metrics = result_columns

# Section 1: Data Overview
st.header("Data Overview")
if st.checkbox("Show Raw Data"):
    st.write(df)

# Section 2: Compare Configurations
st.header("Compare Configurations")

config_id_1 = st.selectbox("Config ID 1", df["tunable_config_id"].unique())
config_id_2 = st.selectbox("Config ID 2", df["tunable_config_id"].unique())
metric = st.selectbox("Metric to Compare", metrics)

if st.button("Compare Configurations"):
    config_1_data = df[df["tunable_config_id"] == config_id_1]
    config_2_data = df[df["tunable_config_id"] == config_id_2]

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        data=config_1_data,
        x="trial_id",
        y=metric,
        marker="o",
        label=f"Config {config_id_1}",
        ax=ax,
    )
    sns.lineplot(
        data=config_2_data,
        x="trial_id",
        y=metric,
        marker="o",
        label=f"Config {config_id_2}",
        ax=ax,
    )

    ax.set_title(f"Comparison of {metric}")
    ax.set_xlabel("Trial ID")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid()

    st.pyplot(fig)

# Section 3: Pair Plot
st.header("Pair Plot for Configurations")
selected_columns = st.multiselect("Select Columns for Pair Plot", config_columns + result_columns)

if st.button("Generate Pair Plot") and selected_columns:
    fig = sns.pairplot(df[selected_columns])
    st.pyplot(fig)

# Section 4: Heatmap
st.header("Correlation Heatmap")
corr_method = st.radio("Correlation Method", ["pearson", "kendall", "spearman"])

if st.button("Generate Heatmap"):
    corr_matrix = df[selected_columns].corr(method=corr_method)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# streamlit run app.py --server.port 8501 --server.address 0.0.0.0
