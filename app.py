# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

sns.set(style="whitegrid")
st.set_page_config(layout="wide", page_title="Internship Feedback Dashboard")

@st.cache_data
def load_data():
    # Load all required inputs; persona is in providers
    df = pd.read_excel("cleaned_feedback_preprocessed.xlsx", dtype=str)
    absa = pd.read_excel("absa_aspect_sentiment.xlsx")
    gap  = pd.read_excel("gap_analysis_metrics_tfidf.xlsx")
    providers = pd.read_excel("provider_scores_and_features.xlsx", index_col="provider_id")
    return df, absa, gap, providers

df, absa_df, gap_df, providers = load_data()

st.title("Internship Feedback Analytics & Recommendations")
tabs = st.tabs([
    "Component 1: ABSA",
    "Component 2: Gap Analysis",
    "Component 3: Personas",
    "Component 4: Recommender"
])

# --- Component 1: ABSA ---
with tabs[0]:
    st.header("Aspect‑Based Sentiment Analysis")
    col1, col2 = st.columns(2)
    with col1:
        counts = absa_df['category'].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="Blues_d")
        ax.set_xlabel("Mentions"); ax.set_ylabel("Aspect")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(data=absa_df, y='category', hue='polarity', ax=ax,
                      order=counts.index, palette="vlag")
        ax.set_xlabel("Count"); ax.set_ylabel("Aspect")
        st.pyplot(fig)

# --- Component 2: Gap Analysis ---
with tabs[1]:
    st.header("Expectation vs. Experience Gap Analysis")
    fig, ax = plt.subplots(figsize=(8,4))
    order = gap_df.groupby('aspect')['hybrid_gap'].median().sort_values().index
    sns.boxplot(data=gap_df, x='aspect', y='hybrid_gap', order=order, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_xlabel("Aspect"); ax.set_ylabel("Hybrid Gap")
    st.pyplot(fig)
    if 'overall_satisfaction' in gap_df.columns:
        corr = gap_df['hybrid_gap'].astype(float).corr(
            gap_df['overall_satisfaction'].astype(float))
        st.markdown(f"**Correlation** between gap and satisfaction: **{corr:.2f}**")

# --- Component 3: Persona Clustering ---
with tabs[2]:
    st.header("Learner Personas")
    # Ensure 'persona' column exists
    if 'persona' not in providers.columns:
        st.error("No 'persona' column found in providers data.")
    else:
        X = providers.drop(columns=['recommendation_score'], errors='ignore').values
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X2 = tsne.fit_transform(X)
        fig, ax = plt.subplots(figsize=(6,6))
        scatter = ax.scatter(
            X2[:,0], X2[:,1],
            c=providers['persona'], cmap='tab10', alpha=0.7
        )
        legend = ax.legend(*scatter.legend_elements(), title="Persona")
        ax.add_artist(legend)
        st.pyplot(fig)
        st.markdown("**Persona counts:**")
        st.bar_chart(providers['persona'].value_counts().sort_index())

# --- Component 4: Context‑Aware Recommender ---
with tabs[3]:
    st.header("Context‑Aware Internship Recommendations")
    st.sidebar.header("Aspect Importance Weights")
    default_weights = {
        'mentorship': 0.2,
        'workload': 0.1,
        'learning_opportunities': 0.25,
        'environment': 0.1,
        'team_collaboration': 0.15,
        'professional_networking': 0.1,
        'career_guidance': 0.1
    }
    weights = {}
    for asp in default_weights:
        weights[asp] = st.sidebar.slider(
            asp.replace("_", " ").title(),
            0.0, 1.0, default_weights[asp], step=0.05
        )
    # Normalize
    total = sum(weights.values())
    weights = {k:v/total for k,v in weights.items()}

    # Prepare and normalize features
    sent_cols = [f"sent_{asp}" for asp in weights if f"sent_{asp}" in providers.columns]
    gap_cols  = [f"gap_{asp}"  for asp in weights if f"gap_{asp}" in providers.columns]
    feat2 = providers.copy()
    scaler_s = MinMaxScaler(); feat2[sent_cols] = scaler_s.fit_transform(feat2[sent_cols])
    scaler_g = MinMaxScaler(); feat2[gap_cols] = 1 - scaler_g.fit_transform(feat2[gap_cols])

    # Compute scores
    combined = sent_cols + gap_cols
    w_list = [weights[asp] for asp in weights if f"sent_{asp}" in sent_cols] + \
             [weights[asp] for asp in weights if f"gap_{asp}" in gap_cols]
    feat2['score'] = feat2[combined].values.dot(np.array(w_list))

    topn = st.number_input("How many to recommend?", min_value=1, max_value=20, value=5)
    recs = feat2['score'].nlargest(topn)
    st.subheader("Top Recommendations")
    st.table(recs.rename("Recommendation Score"))

    st.subheader("Top-3 Explanation")
    for rid in recs.head(3).index:
        st.markdown(f"**ID {rid}** – Score: {feat2.at[rid,'score']:.2f}")
        for asp in weights:
            if f"sent_{asp}" in feat2.columns and f"gap_{asp}" in feat2.columns:
                contrib = weights[asp] * (feat2.at[rid,f"sent_{asp}"] + feat2.at[rid,f"gap_{asp}"])
                st.markdown(f"- {asp.replace('_',' ').title()}: contrib {contrib:.2f}")


