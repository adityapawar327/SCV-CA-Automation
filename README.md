# SVC CA Automation Tools

Automation tools for SVC, a Chartered Accountancy firm, to streamline BRSR reporting and document comparison workflows.

## Features

### 📊 BRSR Principle 6 Updater
Automatically updates environmental sustainability data (energy, water, emissions, waste) in BRSR documents from Excel data.

### 📁 Comparison Tools
- **Excel File Comparator** - Compare up to 15 Excel files with color-coded differences
- **Word Document Comparator** - Compare Word documents with precise visual highlighting
- **PDF & Image Comparator** - Compare images and PDFs using OCR with visual annotations

## Installation

```bash
git clone https://github.com/adityapawar327/SCV-CA-Automation.git
cd SCV-CA-Automation
pip install -r requirements.txt
streamlit run app.py
```

## Usage

1. Select a tool from the sidebar
2. Upload your files
3. Click compare/process
4. Download results

## Tech Stack

Streamlit, Pandas, python-docx, EasyOCR, PyMuPDF, OpenCV

## About

Built for SVC Chartered Accountants to automate repetitive document processing and comparison tasks.