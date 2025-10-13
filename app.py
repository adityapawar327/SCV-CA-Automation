import streamlit as st

# Page configuration
st.set_page_config(
    page_title="BRSR Management System", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("🏢 BRSR Management System")
st.sidebar.markdown("---")

# Feature selection
selected_tool = st.sidebar.selectbox(
    "Select Tool:",
    ["Home", "Principle 6 Updater"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Available Tools")
st.sidebar.markdown("🏠 **Home** - Instructions and overview")
st.sidebar.markdown("📊 **Principle 6 Updater** - Update environmental data")

# Main content based on selection
if selected_tool == "Home":
    st.title("🏢 BRSR Management System")
    st.markdown("### Welcome to the Business Responsibility and Sustainability Reporting (BRSR) Management System")
    st.markdown("---")
    
    # Instructions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 How to Use")
        st.markdown("""
        **Step 1:** Select a tool from the sidebar
        
        **Step 2:** Upload your files as required
        
        **Step 3:** Process and download updated documents
        
        **Step 4:** Review the changes in your BRSR document
        """)
        
    with col2:
        st.subheader("🛠️ Available Tools")
        st.info("**Principle 6 Updater**\nAutomatically updates environmental sustainability data in your BRSR document")
    
    st.markdown("---")
    st.subheader("📊 About Principle 6")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Indicator 1", "Energy Consumption", "🔋")
        
    with col2:
        st.metric("Indicator 3", "Water Consumption", "💧")
        
    with col3:
        st.metric("Indicator 7", "GHG Emissions", "🌍")
        
    with col4:
        st.metric("Indicator 9", "Waste Management", "♻️")
    
    st.markdown("---")
    st.markdown("### 📁 File Requirements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **BRSR Document (.docx)**
        - Must contain Principle 6 tables
        - Tables should have standard BRSR format
        - Ensure proper table structure
        """)
        
    with col2:
        st.markdown("""
        **Excel Data (.xlsx/.xls)**
        - Sheet names should contain 'Principle 6' or relevant keywords
        - Include 'Admin Data calculations' sheet
        - Data should be properly formatted
        """)

elif selected_tool == "Principle 6 Updater":
    from principle6_updater import run_principle6_updater
    run_principle6_updater()