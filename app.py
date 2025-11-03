import streamlit as st
from principle6_updater import run_principle6_updater
from excel_comparator import run_excel_comparator
from ocr_visual_diff import run_ocr_visual_diff
from word_comparator import run_word_comparator

st.set_page_config(
    page_title="BRSR Management System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("📊 BRSR Tools")
st.sidebar.markdown("---")

# Main tool selection
main_tool = st.sidebar.radio(
    "Select Tool",
    ["Home", "Principle 6 Updater", "Comparison"],
    label_visibility="collapsed"
)

# If Comparison is selected, show dropdown
comparison_tool = None
if main_tool == "Comparison":
    comparison_tool = st.sidebar.selectbox(
        "Select Comparison Tool",
        ["Excel File Comparator", "Word Document Comparator", "PDF & Image Comparator"]
    )

# Main content
if main_tool == "Home":
    st.title("📊 BRSR Management System")
    st.caption("Business Responsibility and Sustainability Reporting")
    
    st.divider()
    
    st.markdown("### Available Tools")
    st.markdown("**Principle 6 Updater** - Automatically updates environmental sustainability data (energy, water, emissions, waste) in your BRSR document from Excel data.")
    st.markdown("**Comparison Tools:**")
    st.markdown("- **Excel File Comparator** - Compare up to 15 Excel files and identify differences across sheets, columns, and values.")
    st.markdown("- **Word Document Comparator** - Compare Word documents with visual highlighting of differences.")
    st.markdown("- **PDF & Image Comparator** - Compare images and PDFs using OCR, with visual annotations showing differences directly on the images.")
    
elif main_tool == "Principle 6 Updater":
    run_principle6_updater()
    
elif main_tool == "Comparison":
    if comparison_tool == "Excel File Comparator":
        run_excel_comparator()
    elif comparison_tool == "Word Document Comparator":
        run_word_comparator()
    elif comparison_tool == "PDF & Image Comparator":
        run_ocr_visual_diff()
