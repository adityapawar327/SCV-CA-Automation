import streamlit as st
from principle6_updater import run_principle6_updater
from excel_comparator import run_excel_comparator
# from ocr_visual_diff import run_ocr_visual_diff  # Temporarily disabled due to torch/torchvision compatibility
from word_comparator import run_word_comparator
from document_metadata_analyzer import run_metadata_analyzer

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
    ["Home", "Principle 6 Updater", "Metadata Analyzer", "Comparison"],
    label_visibility="collapsed"
)

# If Comparison is selected, show dropdown
comparison_tool = None
if main_tool == "Comparison":
    comparison_tool = st.sidebar.selectbox(
        "Select Comparison Tool",
        ["Excel File Comparator", "Word Document Comparator"]
        # "PDF & Image Comparator" temporarily disabled due to torch/torchvision compatibility
    )

# Main content
if main_tool == "Home":
    st.title("📊 BRSR Management System")
    st.caption("Business Responsibility and Sustainability Reporting")
    
    st.divider()
    
    st.markdown("### Available Tools")
    
    st.markdown("#### 📋 Principle 6 Updater")
    st.markdown("Automatically updates environmental sustainability data (energy, water, emissions, waste) in your BRSR document from Excel data.")
    
    st.markdown("#### 📄 Document Metadata Analyzer")
    st.markdown("Extract and analyze metadata from multiple documents with powerful features:")
    st.markdown("- **Supported Formats:** PDF, DOCX, XLSX, XLS, CSV, Images (JPG, PNG, GIF, etc.), and more")
    st.markdown("- **Smart Sorting:** Sort files by modification date, creation date, or file size")
    st.markdown("- **Detailed Information:** View file properties, author, creation/modification times, page counts, and more")
    st.markdown("- **Time & Date Display:** Clear format showing ⏰ Time: HH:MM | 📅 Date: DD MMM YYYY")
    st.markdown("- **Quick Access:** Download/open any file directly from the interface")
    st.markdown("- **Export:** Export all metadata to CSV for further analysis")
    st.markdown("- **Perfect for:** Organizing Gmail attachments, managing document libraries, tracking file versions")
    
    st.markdown("#### 🔍 Comparison Tools")
    st.markdown("- **Excel File Comparator** - Compare up to 15 Excel files and identify differences across sheets, columns, and values.")
    st.markdown("- **Word Document Comparator** - Compare Word documents with visual highlighting of differences.")
    # st.markdown("- **PDF & Image Comparator** - Compare images and PDFs using OCR, with visual annotations showing differences directly on the images.")
    
elif main_tool == "Principle 6 Updater":
    run_principle6_updater()

elif main_tool == "Metadata Analyzer":
    run_metadata_analyzer()
    
elif main_tool == "Comparison":
    if comparison_tool == "Excel File Comparator":
        run_excel_comparator()
    elif comparison_tool == "Word Document Comparator":
        run_word_comparator()
    # elif comparison_tool == "PDF & Image Comparator":
    #     run_ocr_visual_diff()
