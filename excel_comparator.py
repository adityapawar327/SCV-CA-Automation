import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

def load_excel_file(file, file_name):
    """Load an Excel file and return all sheets as a dictionary"""
    try:
        xl = pd.ExcelFile(file)
        sheets = {}
        for sheet_name in xl.sheet_names:
            df = pd.read_excel(file, sheet_name=sheet_name)
            sheets[sheet_name] = df
        return sheets, xl.sheet_names
    except Exception as e:
        st.error(f"Error loading {file_name}: {str(e)}")
        return None, None

def highlight_differences(row, diff_positions, file_idx):
    """Apply color highlighting to cells with differences"""
    colors = [
        'background-color: #FF6B6B; color: #000000',  # Bright red
        'background-color: #4ECDC4; color: #000000',  # Bright teal
        'background-color: #FFE66D; color: #000000',  # Bright yellow
        'background-color: #95E1D3; color: #000000',  # Bright mint
        'background-color: #FF9FF3; color: #000000',  # Bright pink
        'background-color: #FECA57; color: #000000',  # Bright orange
        'background-color: #48DBFB; color: #000000',  # Bright cyan
        'background-color: #FF6348; color: #000000',  # Bright coral
        'background-color: #1DD1A1; color: #000000',  # Bright green
        'background-color: #A29BFE; color: #000000',  # Bright purple
        'background-color: #FD79A8; color: #000000',  # Bright rose
        'background-color: #FDCB6E; color: #000000',  # Bright gold
        'background-color: #6C5CE7; color: #FFFFFF',  # Bright indigo
        'background-color: #00B894; color: #000000',  # Bright emerald
        'background-color: #E17055; color: #000000'   # Bright terracotta
    ]
    
    styles = [''] * len(row)
    for col_name, row_idx, files in diff_positions:
        if row.name == row_idx and col_name in row.index:
            if file_idx in files:
                styles[row.index.get_loc(col_name)] = colors[file_idx % len(colors)]
    return styles

def compare_sheet_across_files(sheet_dfs, selected_files=None):
    """Compare the same sheet across multiple files"""
    if selected_files:
        sheet_dfs = {k: v for k, v in sheet_dfs.items() if k in selected_files}
    
    result = {
        'files': list(sheet_dfs.keys()),
        'dataframes': sheet_dfs,
        'shapes': {},
        'differences': [],
        'diff_positions': [],
        'common_columns': None,
        'missing_columns': {}
    }
    
    for file_name, df in sheet_dfs.items():
        result['shapes'][file_name] = df.shape
    
    all_columns = [set(df.columns) for df in sheet_dfs.values()]
    result['common_columns'] = set.intersection(*all_columns) if all_columns else set()
    
    for file_name, df in sheet_dfs.items():
        missing = set(df.columns) - result['common_columns']
        if missing:
            result['missing_columns'][file_name] = list(missing)
    
    if result['common_columns'] and len(sheet_dfs) > 1:
        base_file = result['files'][0]
        base_df = sheet_dfs[base_file]
        
        for file_name in result['files'][1:]:
            compare_df = sheet_dfs[file_name]
            min_rows = min(len(base_df), len(compare_df))
            
            for col in result['common_columns']:
                if col in base_df.columns and col in compare_df.columns:
                    for idx in range(min_rows):
                        base_val = base_df[col].iloc[idx]
                        compare_val = compare_df[col].iloc[idx]
                        
                        if pd.isna(base_val) and pd.isna(compare_val):
                            continue
                        elif pd.isna(base_val) or pd.isna(compare_val) or base_val != compare_val:
                            result['differences'].append({
                                'Sheet': '',
                                'Column': col,
                                'Row': idx,
                                'Base File': base_file,
                                'Base Value': base_val,
                                'Compare File': file_name,
                                'Compare Value': compare_val
                            })
                            result['diff_positions'].append((col, idx, [0, result['files'].index(file_name)]))
    
    return result

def run_excel_comparator():
    """Main function to run the Excel comparator"""
    st.title("📊 Excel File Comparator")
    
    # Initialize session state
    if 'all_files_data' not in st.session_state:
        st.session_state.all_files_data = {}
    if 'comparison_done' not in st.session_state:
        st.session_state.comparison_done = False
    
    uploaded_files = st.file_uploader(
        "Upload Excel files (up to 15)",
        type=['xlsx', 'xls'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if len(uploaded_files) > 15:
            st.warning(f"Maximum 15 files allowed. Processing first 15 files.")
            uploaded_files = uploaded_files[:15]
        
        st.info(f"Loaded {len(uploaded_files)} file(s)")
        
        # Load all files only if not already loaded or files changed
        file_names = [f.name for f in uploaded_files]
        if set(file_names) != set(st.session_state.all_files_data.keys()):
            all_files_data = {}
            all_sheet_names = set()
            
            for file in uploaded_files:
                sheets, sheet_names = load_excel_file(file, file.name)
                if sheets:
                    all_files_data[file.name] = sheets
                    all_sheet_names.update(sheet_names)
            
            st.session_state.all_files_data = all_files_data
            st.session_state.all_sheet_names = all_sheet_names
            st.session_state.comparison_done = False
        
        all_files_data = st.session_state.all_files_data
        all_sheet_names = st.session_state.all_sheet_names
        
        if len(all_files_data) < 2:
            st.error("Need at least 2 valid files to compare")
            return
        
        # Sheet selection
        st.markdown("### Select Sheets to Compare")
        selected_sheets = st.multiselect(
            "Choose sheets",
            sorted(list(all_sheet_names)),
            default=sorted(list(all_sheet_names))
        )
        
        if not selected_sheets:
            st.warning("Please select at least one sheet")
            return
        
        # File selection for comparison
        st.markdown("### Select Files to Compare")
        selected_files = st.multiselect(
            "Choose files",
            list(all_files_data.keys()),
            default=list(all_files_data.keys())
        )
        
        if len(selected_files) < 2:
            st.warning("Please select at least 2 files")
            return
        
        if st.button("Compare", type="primary", use_container_width=True):
            st.session_state.comparison_done = True
            st.session_state.selected_sheets = selected_sheets
            st.session_state.selected_files = selected_files
        
        if st.session_state.comparison_done:
            with st.spinner("Comparing..."):
                
                selected_sheets = st.session_state.selected_sheets
                selected_files = st.session_state.selected_files
                
                # Tabs for each sheet
                tabs = st.tabs(selected_sheets + ["Summary"])
                
                all_differences = []
                
                for idx, sheet_name in enumerate(selected_sheets):
                    with tabs[idx]:
                        st.markdown(f"### {sheet_name}")
                        
                        # Get dataframes for this sheet
                        sheet_dfs = {}
                        for file_name in selected_files:
                            if sheet_name in st.session_state.all_files_data[file_name]:
                                sheet_dfs[file_name] = st.session_state.all_files_data[file_name][sheet_name]
                        
                        if len(sheet_dfs) < 2:
                            st.warning(f"Sheet '{sheet_name}' not found in enough files")
                            continue
                        
                        # Compare
                        result = compare_sheet_across_files(sheet_dfs, selected_files)
                        
                        # Show dimensions
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.markdown("**Dimensions:**")
                            for file, shape in result['shapes'].items():
                                st.text(f"{file}: {shape[0]}×{shape[1]}")
                        
                        with col2:
                            if result['differences']:
                                st.markdown(f"**Differences: {len(result['differences'])}**")
                            else:
                                st.success("No differences found")
                        
                        # Show each file's data with highlighting
                        st.markdown("---")
                        file_tabs = st.tabs(result['files'])
                        
                        for file_idx, file_name in enumerate(result['files']):
                            with file_tabs[file_idx]:
                                # Toggle between view and edit mode
                                mode_key = f"mode_{sheet_name}_{file_name}"
                                if mode_key not in st.session_state:
                                    st.session_state[mode_key] = "view"
                                
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    mode = st.radio(
                                        "Mode",
                                        ["View (with highlights)", "Edit"],
                                        key=f"radio_{mode_key}",
                                        horizontal=True
                                    )
                                
                                # Get current data from session state
                                current_df = st.session_state.all_files_data[file_name][sheet_name]
                                
                                if mode == "View (with highlights)":
                                    # Show with color highlighting
                                    if result['diff_positions']:
                                        styled_df = current_df.style.apply(
                                            lambda row: highlight_differences(row, result['diff_positions'], file_idx),
                                            axis=1
                                        )
                                        st.dataframe(styled_df, use_container_width=True, height=400)
                                    else:
                                        st.dataframe(current_df, use_container_width=True, height=400)
                                else:
                                    # Editable mode
                                    editor_key = f"editor_{sheet_name}_{file_name}"
                                    edited_df = st.data_editor(
                                        current_df,
                                        use_container_width=True,
                                        height=400,
                                        num_rows="dynamic",
                                        key=editor_key
                                    )
                                    # Update the dataframe in session state
                                    st.session_state.all_files_data[file_name][sheet_name] = edited_df
                                
                                # Download edited file
                                output = BytesIO()
                                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                    for sname, sdf in st.session_state.all_files_data[file_name].items():
                                        sdf.to_excel(writer, sheet_name=sname, index=False)
                                output.seek(0)
                                
                                st.download_button(
                                    label=f"Download {file_name}",
                                    data=output,
                                    file_name=f"edited_{file_name}",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key=f"download_{sheet_name}_{file_name}"
                                )
                        
                        # Add to summary
                        if result['differences']:
                            for diff in result['differences']:
                                diff['Sheet'] = sheet_name
                                all_differences.append(diff)
                
                # Summary tab
                with tabs[-1]:
                    st.markdown("### Comparison Summary")
                    
                    if all_differences:
                        st.markdown(f"**Total Differences Found: {len(all_differences)}**")
                        
                        # Group by sheet
                        diff_by_sheet = {}
                        for diff in all_differences:
                            sheet = diff['Sheet']
                            if sheet not in diff_by_sheet:
                                diff_by_sheet[sheet] = []
                            diff_by_sheet[sheet].append(diff)
                        
                        for sheet, diffs in diff_by_sheet.items():
                            with st.expander(f"{sheet} ({len(diffs)} differences)"):
                                diff_df = pd.DataFrame(diffs)
                                st.dataframe(diff_df, use_container_width=True)
                        
                        # Export all differences
                        export_df = pd.DataFrame(all_differences)
                        csv = export_df.to_csv(index=False)
                        st.download_button(
                            label="Download All Differences (CSV)",
                            data=csv,
                            file_name="all_differences.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.success("No differences found across all selected sheets")
