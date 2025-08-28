import streamlit as st

# Main page config
st.set_page_config(
    page_title="Sales Prediction App",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sales Prediction App")
st.markdown("""
Welcome to the **Sales Prediction Streamlit App**!  

Use the sidebar to navigate through the different sections:
- **Home** → Introduction & overview  
- **Dashboard** → Explore sales data & visualizations  
- **Prediction** → Test the trained models (single input or batch mode)  
""")
