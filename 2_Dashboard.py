# pages/2_Dashboard.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📈 Dashboard — 6 Key Visualizations")

# Load dataset from full path
try:
    df = pd.read_csv(r"E:\ml project\data\sales_data.csv")
    st.info("Loaded dataset from E:\\ml project\\data\\sales_data.csv ✅")
except FileNotFoundError:
    st.error("❌ File not found. Please check the file path: E:\\ml project\\data\\sales_data.csv")
    st.stop()

st.subheader("Data preview")
st.dataframe(df.head())

# 1) Revenue distribution
st.subheader("1) Revenue distribution")
fig1, ax1 = plt.subplots()
sns.histplot(df['revenue'], kde=True, ax=ax1)
ax1.set_xlabel("Revenue")
st.pyplot(fig1)

# 2) nb_sold distribution
if 'nb_sold' in df.columns:
    st.subheader("2) Number of items sold (nb_sold)")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['nb_sold'], kde=True, ax=ax2)
    st.pyplot(fig2)

# 3) sales_method counts
if 'sales_method' in df.columns:
    st.subheader("3) Sales method counts")
    fig3, ax3 = plt.subplots()
    sns.countplot(x='sales_method', data=df, ax=ax3)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=15)
    st.pyplot(fig3)

# 4) correlation heatmap
numeric_cols = [c for c in ['nb_sold','revenue','years_as_customer','nb_site_visits'] if c in df.columns]
if len(numeric_cols) >= 2:
    st.subheader("4) Correlation heatmap")
    fig4, ax4 = plt.subplots(figsize=(6,5))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax4)
    st.pyplot(fig4)

# 5) total revenue per week
if 'week' in df.columns:
    st.subheader("5) Total revenue per week (sum)")
    fig5, ax5 = plt.subplots()
    df.groupby("week")["revenue"].sum().plot(kind="bar", ax=ax5)
    ax5.set_ylabel("Total revenue")
    st.pyplot(fig5)

# 6) average revenue by sales method
if 'sales_method' in df.columns:
    st.subheader("6) Average revenue by sales method")
    fig6, ax6 = plt.subplots()
    sns.barplot(x='sales_method', y='revenue', data=df, estimator='mean', ax=ax6)
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=15)
    st.pyplot(fig6)
