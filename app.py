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
    df     = pd.read_excel("cleaned_feedback_preprocessed.xlsx", dtype=str)
    absa   = pd.read_excel("absa_aspect_sentiment.xlsx")
    gap    = pd.read_excel("gap_analysis_metrics_tfidf.xlsx")
    prov   = pd.read_excel("provider_scores_and_features.xlsx", dtype=float)
    if 'id' in prov.columns:
        prov.set_index('id', inplace=True)
    return df, absa, gap, prov

df, absa_df, gap_df, providers = load_data()

st.title("Internship Feedback Analytics & Recommendations")
tabs = st.tabs([
    "Component 1: ABSA",
    "Component 2: Gap Analysis",
    "Component 3: Persona & Recommendations"
])

# Component 1: ABSA
with tabs[0]:
    st.header("Aspect‑Based Sentiment Analysis")
    counts = absa_df['aspect_category'].value_counts()
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        sns.barplot(x=counts.values, y=counts.index, palette="Blues_d", ax=ax)
        ax.set(xlabel="Mentions", ylabel="Aspect")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(
            data=absa_df,
            y='aspect_category',
            hue='polarity',
            order=counts.index,
            palette="vlag",
            ax=ax
        )
        ax.set(xlabel="Count", ylabel="Aspect")
        st.pyplot(fig)

# --- Component 2: Enhanced Gap Analysis Visuals ---
with tabs[1]:
    st.header("Expectation vs. Experience Gap Analysis")

    # 2.1 Median Hybrid Gap per Aspect (Bar Chart)
    st.subheader("Median Hybrid Gap by Aspect")
    median_gaps = gap_df.groupby('aspect')['hybrid_gap'] \
                        .median() \
                        .sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=median_gaps.values,
        y=median_gaps.index,
        palette="magma",
        ax=ax
    )
    ax.set(xlabel="Median Hybrid Gap", ylabel="Aspect")
    ax.set_title("Median Expectation–Experience Gap per Aspect")
    st.pyplot(fig)

    # 2.2 Distribution of Hybrid Gaps (Violin Plot)
    st.subheader("Gap Distribution per Aspect")
    aspect_order = median_gaps.index  # keep same order
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.violinplot(
        data=gap_df,
        x='hybrid_gap',
        y='aspect',
        order=aspect_order,
        scale='width',
        inner='quartile',
        palette="coolwarm",
        ax=ax2
    )
    ax2.set(xlabel="Hybrid Gap Score", ylabel="Aspect")
    ax2.set_title("Distribution of Expectation–Experience Gaps")
    st.pyplot(fig2)

    # 2.3 Correlation Scatter (if overall_satisfaction exists)
    if 'overall_satisfaction' in gap_df.columns:
        st.subheader("Gap vs. Overall Satisfaction")
        corr = gap_df['hybrid_gap'].astype(float).corr(
            gap_df['overall_satisfaction'].astype(float)
        )
        st.markdown(f"**Pearson correlation:** {corr:.2f}")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=gap_df,
            x='hybrid_gap',
            y='overall_satisfaction',
            alpha=0.6,
            ax=ax3
        )
        ax3.set(xlabel="Hybrid Gap", ylabel="Overall Satisfaction")
        ax3.set_title("Hybrid Gap vs. Satisfaction")
        st.pyplot(fig3)


# Component 3: Persona Clustering & Per-Record Recommendation
with tabs[2]:
    st.header("Learner Persona Clustering & Recommendations")

    # 3.1 Persona Clustering (t-SNE)
    if 'persona' in providers.columns:
        st.subheader("Persona Clustering (t‑SNE)")
        feats = providers.drop(columns=['recommendation_score'], errors='ignore').values
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embed = tsne.fit_transform(feats)
        fig, ax = plt.subplots(figsize=(6,6))
        sc = ax.scatter(embed[:,0], embed[:,1],
                        c=providers['persona'], cmap='tab10', alpha=0.7)
        ax.legend(*sc.legend_elements(), title="Persona")
        ax.set(title="t‑SNE Projection of Personas", xlabel="t‑SNE 1", ylabel="t‑SNE 2")
        st.pyplot(fig)

        st.subheader("Persona Distribution")
        st.bar_chart(providers['persona'].value_counts().sort_index())
    else:
        st.warning("No 'persona' column found—skipping clustering.")

    st.subheader("Select a Record to View Recommendation")
    # 3.2 Dropdown for record selection
    record_id = st.selectbox("Record ID", providers.index.tolist())

    # 3.3 Show record's features
    st.markdown(f"**Features for Record {record_id}:**")
    st.write(providers.loc[record_id])

    # 3.4 Compute this record’s recommendation score & top aspects
    # Define default weights (same as your model)
    default_weights = {
        'mentorship': 0.20,
        'workload': 0.10,
        'learning_opportunities': 0.25,
        'environment': 0.10,
        'team_collaboration': 0.15,
        'professional_networking': 0.10,
        'career_guidance': 0.10
    }
    # Let user adjust via sidebar
    st.sidebar.header("Aspect Weights for Recommendation")
    weights = {
        asp: st.sidebar.slider(
            asp.replace("_"," ").title(), 0.0, 1.0, default_weights[asp], step=0.05
        ) for asp in default_weights
    }
    # normalize
    total = sum(weights.values()) or 1
    weights = {k: v/total for k,v in weights.items()}

    # Prepare scaled features for this record
    sent_cols = [f"sent_{asp}" for asp in weights if f"sent_{asp}" in providers.columns]
    gap_cols  = [f"gap_{asp}"  for asp in weights if f"gap_{asp}" in providers.columns]
    row = providers.loc[record_id].copy()

    # scale record features relative to entire dataset
    scaler_s = MinMaxScaler().fit(providers[sent_cols])
    scaler_g = MinMaxScaler().fit(providers[gap_cols])
    s_vals = scaler_s.transform([row[sent_cols]])[0]
    g_vals = 1.0 - scaler_g.transform([row[gap_cols]])[0]

    # compute score
    combined = sent_cols + gap_cols
    w_list = [weights[asp] for asp in weights if f"sent_{asp}" in sent_cols] + \
             [weights[asp] for asp in weights if f"gap_{asp}" in gap_cols]
    score = np.dot(np.concatenate([s_vals, g_vals]), np.array(w_list))

    st.subheader(f"Recommendation Score: {score:.2f}")

    # 3.5 Show top-3 aspects for this record
    contribs = {}
    for asp in weights:
        sc_col, gp_col = f"sent_{asp}", f"gap_{asp}"
        if sc_col in providers.columns and gp_col in providers.columns:
            val_s = s_vals[sent_cols.index(sc_col)]
            val_g = g_vals[gap_cols.index(gp_col)]
            contribs[asp] = weights[asp] * (val_s + val_g)
    top_asps = sorted(contribs.items(), key=lambda x: x[1], reverse=True)[:3]

    st.subheader("Top Contributing Aspects")
    for asp, c in top_asps:
        st.markdown(f"- **{asp.replace('_',' ').title()}**: contribution {c:.2f}")
