import streamlit as st
import pandas as pd
from docx import Document
from io import BytesIO
import numpy as np

def extract_principle6_data(excel_file):
    """Extract Principle 6 data from Excel file"""
    try:
        xl = pd.ExcelFile(excel_file)
        principle6_data = {}
        
        for sheet_name in xl.sheet_names:
            lower_name = sheet_name.lower()
            if ('principle' in lower_name and '6' in lower_name) or 'admin data' in lower_name:
                # Read the sheet with proper headers
                df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
                principle6_data[sheet_name] = df
                st.info(f"Found sheet: {sheet_name} with {len(df)} rows and {len(df.columns)} columns")
                
                # Show a preview of the data structure
                with st.expander(f"Preview of {sheet_name}"):
                    st.dataframe(df.head(10))
        
        return principle6_data
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        return None

def find_table_by_keyword(doc, keywords):
    """Find table in document by searching for keywords"""
    for table in doc.tables:
        table_text = ' '.join([cell.text for row in table.rows for cell in row.cells]).lower()
        if any(keyword.lower() in table_text for keyword in keywords):
            return table
    return None

def safe_float(value):
    """Safely convert value to float"""
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except:
        return 0.0

def extract_principle6_section(admin_data):
    """Extract the Principle 6 section data from the Admin Data sheet"""
    try:
        # Find the "Principle 6" section (around row 21 in the image)
        principle6_start = None
        for idx, row in admin_data.iterrows():
            if pd.notna(row.iloc[0]) and 'principle 6' in str(row.iloc[0]).lower():
                principle6_start = idx
                break
        
        if principle6_start is None:
            st.warning("Could not find 'Principle 6' section in Excel data")
            return None
        
        # Extract data from the Principle 6 section
        principle6_section = admin_data.iloc[principle6_start:principle6_start+20]  # Get next 20 rows
        
        return principle6_section
    except Exception as e:
        st.error(f"Error extracting Principle 6 section: {str(e)}")
        return None

def update_energy_table(doc, admin_data):
    """Update energy consumption table (Indicator 1) from Admin Data calculations sheet"""
    try:
        keywords = ['energy consumption', 'renewable', 'non-renewable', 'from renewable sources']
        table = find_table_by_keyword(doc, keywords)
        
        if not table:
            st.warning("Energy consumption table not found")
            return False
        
        # Find the "Total" row in the Excel data (usually the last row with data)
        total_row_idx = None
        for idx, row in admin_data.iterrows():
            if pd.notna(row.iloc[0]) and str(row.iloc[0]).lower().strip() == 'total':
                total_row_idx = idx
                break
        
        if total_row_idx is None:
            st.warning("Could not find 'Total' row in Excel data")
            return False
        
        # Extract electricity consumption data from Total row
        # Based on the image: columns B, C are Scope 2 (Renewable), D, E are Scope 2 (Non-renewable)
        renewable_fy25 = safe_float(admin_data.iloc[total_row_idx, 1])  # Column B (FY 2024-25)
        renewable_fy24 = safe_float(admin_data.iloc[total_row_idx, 2])  # Column C (FY 2023-24)
        
        non_renewable_fy25 = safe_float(admin_data.iloc[total_row_idx, 3])  # Column D (FY 2024-25)
        non_renewable_fy24 = safe_float(admin_data.iloc[total_row_idx, 4])  # Column E (FY 2023-24)
        
        # Calculate total electricity consumption
        total_electricity_fy25 = renewable_fy25 + non_renewable_fy25
        total_electricity_fy24 = renewable_fy24 + non_renewable_fy24
        
        # Extract fuel consumption data (columns F-M for diesel, PNG, LPG, Petrol)
        diesel_fy25 = safe_float(admin_data.iloc[total_row_idx, 5])  # Column F
        png_fy25 = safe_float(admin_data.iloc[total_row_idx, 6])     # Column G
        lpg_fy25 = safe_float(admin_data.iloc[total_row_idx, 7])     # Column H
        petrol_fy25 = safe_float(admin_data.iloc[total_row_idx, 8])  # Column I
        
        diesel_fy24 = safe_float(admin_data.iloc[total_row_idx, 9])  # Column J
        png_fy24 = safe_float(admin_data.iloc[total_row_idx, 10])    # Column K
        lpg_fy24 = safe_float(admin_data.iloc[total_row_idx, 11])    # Column L
        petrol_fy24 = safe_float(admin_data.iloc[total_row_idx, 12]) # Column M
        
        # Calculate total fuel consumption
        total_fuel_fy25 = diesel_fy25 + png_fy25 + lpg_fy25 + petrol_fy25
        total_fuel_fy24 = diesel_fy24 + png_fy24 + lpg_fy24 + petrol_fy24
        
        # Calculate total energy consumed (electricity + fuel)
        total_energy_fy25 = total_electricity_fy25 + total_fuel_fy25
        total_energy_fy24 = total_electricity_fy24 + total_fuel_fy24
        
        # Update table cells
        for i, row in enumerate(table.rows):
            if i == 0:  # Skip header
                continue
            row_text = row.cells[0].text.lower()
            
            # Total electricity consumption (renewable sources)
            if 'electricity consumption' in row_text and 'renewable' in row_text:
                if len(row.cells) > 1:
                    row.cells[1].text = f"{renewable_fy25:.2f}"
                if len(row.cells) > 2:
                    row.cells[2].text = f"{renewable_fy24:.2f}"
            
            # Total electricity consumption (non-renewable sources)
            elif 'electricity consumption' in row_text and 'non-renewable' in row_text:
                if len(row.cells) > 1:
                    row.cells[1].text = f"{non_renewable_fy25:.2f}"
                if len(row.cells) > 2:
                    row.cells[2].text = f"{non_renewable_fy24:.2f}"
            
            # Total fuel consumption
            elif 'fuel consumption' in row_text:
                if len(row.cells) > 1:
                    row.cells[1].text = f"{total_fuel_fy25:.2f}"
                if len(row.cells) > 2:
                    row.cells[2].text = f"{total_fuel_fy24:.2f}"
            
            # Total energy consumed
            elif 'total energy consumed' in row_text:
                if len(row.cells) > 1:
                    row.cells[1].text = f"{total_energy_fy25:.2f}"
                if len(row.cells) > 2:
                    row.cells[2].text = f"{total_energy_fy24:.2f}"
        
        st.success("✅ Energy consumption table updated")
        st.info(f"Updated with: Renewable: {renewable_fy25:.2f}/{renewable_fy24:.2f}, Non-renewable: {non_renewable_fy25:.2f}/{non_renewable_fy24:.2f}")
        return True
    except Exception as e:
        st.error(f"Error updating energy table: {str(e)}")
        return False

def update_water_table(doc, water_data):
    """Update water consumption table (Indicator 3)"""
    try:
        keywords = ['water withdrawal', 'water consumption', 'third party water']
        table = find_table_by_keyword(doc, keywords)
        
        if not table:
            st.warning("Water consumption table not found")
            return False
        
        # Extract water data from the appropriate sheet
        # Assuming water data is in specific rows/columns
        third_party_fy25 = safe_float(water_data.iloc[2, 1]) if len(water_data) > 2 else 0
        third_party_fy24 = safe_float(water_data.iloc[2, 2]) if len(water_data) > 2 else 0
        
        total_withdrawal_fy25 = safe_float(water_data.iloc[6, 1]) if len(water_data) > 6 else 0
        total_withdrawal_fy24 = safe_float(water_data.iloc[6, 2]) if len(water_data) > 6 else 0
        
        total_consumption_fy25 = safe_float(water_data.iloc[7, 1]) if len(water_data) > 7 else 0
        total_consumption_fy24 = safe_float(water_data.iloc[7, 2]) if len(water_data) > 7 else 0
        
        # Update table
        for row in table.rows:
            row_text = row.cells[0].text.lower()
            
            if 'third party water' in row_text:
                row.cells[1].text = f"{third_party_fy25:.2f}"
                row.cells[2].text = f"{third_party_fy24:.2f}"
            elif 'total volume of water withdrawal' in row_text:
                row.cells[1].text = f"{total_withdrawal_fy25:.2f}"
                row.cells[2].text = f"{total_withdrawal_fy24:.2f}"
            elif 'total volume of water consumption' in row_text and 'kilolitres' in row_text:
                row.cells[1].text = f"{total_consumption_fy25:.2f}"
                row.cells[2].text = f"{total_consumption_fy24:.2f}"
        
        st.success("✅ Water consumption table updated")
        return True
    except Exception as e:
        st.error(f"Error updating water table: {str(e)}")
        return False

def update_emissions_table(doc, emissions_data):
    """Update GHG emissions table (Indicator 7)"""
    try:
        keywords = ['scope 1', 'scope 2', 'ghg emissions', 'greenhouse gas']
        table = find_table_by_keyword(doc, keywords)
        
        if not table:
            st.warning("Emissions table not found")
            return False
        
        # Extract emissions data
        scope1_fy25 = safe_float(emissions_data.iloc[0, 2]) if len(emissions_data) > 0 else 0
        scope1_fy24 = safe_float(emissions_data.iloc[0, 3]) if len(emissions_data) > 0 else 0
        
        scope2_fy25 = safe_float(emissions_data.iloc[1, 2]) if len(emissions_data) > 1 else 0
        scope2_fy24 = safe_float(emissions_data.iloc[1, 3]) if len(emissions_data) > 1 else 0
        
        total_fy25 = scope1_fy25 + scope2_fy25
        total_fy24 = scope1_fy24 + scope2_fy24
        
        # Update table
        for row in table.rows:
            row_text = row.cells[0].text.lower()
            
            if 'total scope 1 emissions' in row_text:
                row.cells[2].text = f"{scope1_fy25:.2f}"
                row.cells[3].text = f"{scope1_fy24:.2f}"
            elif 'total scope 2 emissions' in row_text:
                row.cells[2].text = f"{scope2_fy25:.2f}"
                row.cells[3].text = f"{scope2_fy24:.2f}"
            elif 'total scope 1 and scope 2' in row_text and 'intensity' not in row_text:
                row.cells[2].text = f"{total_fy25:.2f}"
                row.cells[3].text = f"{total_fy24:.2f}"
        
        st.success("✅ Emissions table updated")
        return True
    except Exception as e:
        st.error(f"Error updating emissions table: {str(e)}")
        return False

def update_waste_table(doc, waste_data):
    """Update waste management table (Indicator 9)"""
    try:
        keywords = ['waste generated', 'plastic waste', 'e-waste', 'recycled']
        table = find_table_by_keyword(doc, keywords)
        
        if not table:
            st.warning("Waste management table not found")
            return False
        
        # Extract waste data
        plastic_fy25 = safe_float(waste_data.iloc[1, 1]) if len(waste_data) > 1 else 0
        plastic_fy24 = safe_float(waste_data.iloc[1, 2]) if len(waste_data) > 1 else 0
        
        ewaste_fy25 = safe_float(waste_data.iloc[2, 1]) if len(waste_data) > 2 else 0
        ewaste_fy24 = safe_float(waste_data.iloc[2, 2]) if len(waste_data) > 2 else 0
        
        total_fy25 = safe_float(waste_data.iloc[8, 1]) if len(waste_data) > 8 else 0
        total_fy24 = safe_float(waste_data.iloc[8, 2]) if len(waste_data) > 8 else 0
        
        # Update table
        for row in table.rows:
            row_text = row.cells[0].text.lower()
            
            if 'plastic waste' in row_text and '(a)' in row_text:
                row.cells[1].text = f"{plastic_fy25:.2f}"
                row.cells[2].text = f"{plastic_fy24:.2f}"
            elif 'e-waste' in row_text and '(b)' in row_text:
                row.cells[1].text = f"{ewaste_fy25:.2f}"
                row.cells[2].text = f"{ewaste_fy24:.2f}"
            elif 'total' in row_text and 'a+b+c+d+e+f+g+h' in row_text:
                row.cells[1].text = f"{total_fy25:.2f}"
                row.cells[2].text = f"{total_fy24:.2f}"
        
        st.success("✅ Waste management table updated")
        return True
    except Exception as e:
        st.error(f"Error updating waste table: {str(e)}")
        return False

def run_principle6_updater():
    """Main function to run the Principle 6 updater"""
    st.title("📊 BRSR Principle 6 Data Updater")
    st.markdown("---")

    # File uploaders
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Upload BRSR Document")
        docx_file = st.file_uploader("Choose DOCX file", type=['docx'])

    with col2:
        st.subheader("📊 Upload Excel Data")
        excel_file = st.file_uploader("Choose Excel file with Principle 6 data", type=['xlsx', 'xls'])

    # Process button
    if st.button("🔄 Process and Update Document", type="primary", use_container_width=True):
        if docx_file and excel_file:
            with st.spinner("Processing files..."):
                try:
                    doc = Document(docx_file)
                    principle6_data = extract_principle6_data(excel_file)
                    
                    if principle6_data:
                        st.success(f"✅ Found {len(principle6_data)} data sheet(s)")
                        
                        with st.expander("📋 View Available Data Sheets"):
                            for sheet_name, df in principle6_data.items():
                                st.write(f"**{sheet_name}**")
                                st.dataframe(df.head())
                        
                        progress_bar = st.progress(0)
                        st.info("Updating Principle 6 tables...")
                        
                        # Update energy consumption (Indicator 1)
                        if 'Admin Data calculations' in principle6_data:
                            update_energy_table(doc, principle6_data['Admin Data calculations'])
                        progress_bar.progress(25)
                        
                        # Update water consumption (Indicator 3)
                        for sheet_name in principle6_data.keys():
                            if 'pt 3' in sheet_name.lower() or 'water' in sheet_name.lower():
                                update_water_table(doc, principle6_data[sheet_name])
                                break
                        progress_bar.progress(50)
                        
                        # Update emissions (Indicator 7)
                        for sheet_name in principle6_data.keys():
                            if 'pt 7' in sheet_name.lower() or 'emission' in sheet_name.lower():
                                update_emissions_table(doc, principle6_data[sheet_name])
                                break
                        progress_bar.progress(75)
                        
                        # Update waste management (Indicator 9)
                        for sheet_name in principle6_data.keys():
                            if 'pt 9' in sheet_name.lower() or 'waste' in sheet_name.lower():
                                update_waste_table(doc, principle6_data[sheet_name])
                                break
                        progress_bar.progress(100)
                        
                        # Save updated document
                        output = BytesIO()
                        doc.save(output)
                        output.seek(0)
                        
                        st.success("✅ Document updated successfully!")
                        
                        st.download_button(
                            label="📥 Download Updated BRSR Document",
                            data=output,
                            file_name="Updated_BRSR_Principle6.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )
                    else:
                        st.error("❌ No Principle 6 data found in Excel file")
                        
                except Exception as e:
                    st.error(f"❌ Error processing files: {str(e)}")
        else:
            st.warning("⚠️ Please upload both DOCX and Excel files")