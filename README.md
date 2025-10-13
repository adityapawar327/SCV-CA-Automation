# BRSR Principle 6 Updater

A Streamlit application for automatically updating Business Responsibility and Sustainability Reporting (BRSR) Principle 6 environmental data in Word documents.

## Features

- 📊 **Automated Data Updates** - Extract data from Excel and update BRSR documents
- 🏠 **User-friendly Interface** - Clean web interface with sidebar navigation
- 📄 **Document Processing** - Handles .docx BRSR documents and .xlsx/.xls Excel files
- 🔋 **Energy Consumption** - Updates Indicator 1 (Energy consumption data)
- 💧 **Water Consumption** - Updates Indicator 3 (Water consumption data)
- 🌍 **GHG Emissions** - Updates Indicator 7 (Greenhouse gas emissions)
- ♻️ **Waste Management** - Updates Indicator 9 (Waste management data)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/adityapawar327/SCV-CA-Automation.git
cd SCV-CA-Automation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. **Select Tool**: Choose "Principle 6 Updater" from the sidebar
2. **Upload Files**: 
   - Upload your BRSR document (.docx)
   - Upload Excel file with environmental data (.xlsx/.xls)
3. **Process**: Click "Process and Update Document"
4. **Download**: Download the updated BRSR document

## File Requirements

### BRSR Document (.docx)
- Must contain Principle 6 tables
- Tables should have standard BRSR format
- Ensure proper table structure

### Excel Data (.xlsx/.xls)
- Sheet named "Admin Data calculations" 
- Data structure with office locations and environmental metrics
- Columns for electricity consumption, fuel consumption, etc.

## Supported Indicators

- **Indicator 1**: Energy Consumption (Renewable/Non-renewable)
- **Indicator 3**: Water Consumption and Withdrawal
- **Indicator 7**: Scope 1 and Scope 2 GHG Emissions
- **Indicator 9**: Waste Management (Plastic, E-waste, etc.)

## Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Document Handling**: python-docx
- **Excel Processing**: openpyxl, xlrd

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Contact

For questions or support, please contact the development team.