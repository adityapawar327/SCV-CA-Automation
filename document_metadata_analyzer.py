import streamlit as st
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from docx import Document
import openpyxl
from PIL import Image
import fitz  # PyMuPDF
import json

class DocumentMetadataAnalyzer:
    """Extract and analyze metadata from various document formats"""
    
    SUPPORTED_FORMATS = {
        'documents': ['.pdf', '.docx', '.doc', '.txt'],
        'spreadsheets': ['.xlsx', '.xls', '.csv'],
        'presentations': ['.pptx', '.ppt'],
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'],
        'archives': ['.zip', '.rar', '.7z'],
        'other': ['.msg', '.eml']
    }
    
    def __init__(self):
        self.metadata_cache = []
    
    def extract_pdf_metadata(self, file_path):
        """Extract metadata from PDF files"""
        try:
            doc = fitz.open(file_path)
            metadata = doc.metadata
            
            return {
                'title': metadata.get('title', 'N/A'),
                'author': metadata.get('author', 'N/A'),
                'subject': metadata.get('subject', 'N/A'),
                'creator': metadata.get('creator', 'N/A'),
                'producer': metadata.get('producer', 'N/A'),
                'creation_date': metadata.get('creationDate', 'N/A'),
                'modification_date': metadata.get('modDate', 'N/A'),
                'pages': doc.page_count,
                'encrypted': doc.is_encrypted
            }
        except Exception as e:
            return {'error': str(e)}
    
    def extract_docx_metadata(self, file_path):
        """Extract metadata from DOCX files"""
        try:
            doc = Document(file_path)
            core_props = doc.core_properties
            
            return {
                'title': core_props.title or 'N/A',
                'author': core_props.author or 'N/A',
                'subject': core_props.subject or 'N/A',
                'keywords': core_props.keywords or 'N/A',
                'comments': core_props.comments or 'N/A',
                'created': core_props.created.isoformat() if core_props.created else 'N/A',
                'modified': core_props.modified.isoformat() if core_props.modified else 'N/A',
                'last_modified_by': core_props.last_modified_by or 'N/A',
                'revision': core_props.revision or 'N/A',
                'paragraphs': len(doc.paragraphs),
                'tables': len(doc.tables)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def extract_xlsx_metadata(self, file_path):
        """Extract metadata from XLSX files"""
        try:
            wb = openpyxl.load_workbook(file_path, read_only=True)
            props = wb.properties
            
            return {
                'title': props.title or 'N/A',
                'author': props.creator or 'N/A',
                'subject': props.subject or 'N/A',
                'keywords': props.keywords or 'N/A',
                'comments': props.description or 'N/A',
                'created': props.created.isoformat() if props.created else 'N/A',
                'modified': props.modified.isoformat() if props.modified else 'N/A',
                'last_modified_by': props.lastModifiedBy or 'N/A',
                'sheets': len(wb.sheetnames),
                'sheet_names': ', '.join(wb.sheetnames)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def extract_image_metadata(self, file_path):
        """Extract metadata from image files"""
        try:
            img = Image.open(file_path)
            exif_data = img._getexif() if hasattr(img, '_getexif') else None
            
            metadata = {
                'format': img.format,
                'mode': img.mode,
                'size': f"{img.size[0]}x{img.size[1]}",
                'width': img.size[0],
                'height': img.size[1]
            }
            
            if exif_data:
                metadata['exif_available'] = True
            
            return metadata
        except Exception as e:
            return {'error': str(e)}

    def extract_file_system_metadata(self, file_path):
        """Extract file system metadata"""
        try:
            stat = os.stat(file_path)
            return {
                'size_bytes': stat.st_size,
                'size_readable': self.format_file_size(stat.st_size),
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'accessed': datetime.fromtimestamp(stat.st_atime).isoformat()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def format_file_size(self, size_bytes):
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
    
    def format_datetime(self, date_str):
        """Format datetime to '23:07 23 Nov 2025' format"""
        if date_str == 'N/A' or date_str is None:
            return 'N/A'
        try:
            # Handle different date formats
            if 'D:' in str(date_str):  # PDF date format
                date_str = date_str.replace('D:', '').split('+')[0].split('-')[0]
                dt = datetime.strptime(date_str[:14], '%Y%m%d%H%M%S')
            else:
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            
            # Format as "23:07 23 Nov 2025"
            return dt.strftime('%H:%M %d %b %Y')
        except:
            return date_str
    
    def extract_metadata(self, file_path):
        """Extract metadata based on file type"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        metadata = {
            'filename': file_path.name,
            'extension': extension,
            'path': str(file_path.absolute())
        }
        
        # Add file system metadata
        fs_metadata = self.extract_file_system_metadata(file_path)
        metadata.update(fs_metadata)
        
        # Extract format-specific metadata
        if extension == '.pdf':
            metadata['type'] = 'PDF Document'
            metadata.update(self.extract_pdf_metadata(file_path))
        elif extension in ['.docx', '.doc']:
            metadata['type'] = 'Word Document'
            if extension == '.docx':
                metadata.update(self.extract_docx_metadata(file_path))
        elif extension in ['.xlsx', '.xls']:
            metadata['type'] = 'Excel Spreadsheet'
            if extension == '.xlsx':
                metadata.update(self.extract_xlsx_metadata(file_path))
        elif extension in self.SUPPORTED_FORMATS['images']:
            metadata['type'] = 'Image'
            metadata.update(self.extract_image_metadata(file_path))
        else:
            metadata['type'] = 'Other'
        
        return metadata
    
    def analyze_multiple_files(self, file_paths):
        """Analyze multiple files and return sorted metadata"""
        results = []
        
        for file_path in file_paths:
            try:
                metadata = self.extract_metadata(file_path)
                results.append(metadata)
            except Exception as e:
                st.error(f"Error processing {file_path}: {str(e)}")
        
        return results
    
    def sort_by_date(self, metadata_list, date_field='modified', ascending=False):
        """Sort metadata by date field"""
        def get_date(item):
            date_str = item.get(date_field, 'N/A')
            if date_str == 'N/A' or date_str is None:
                return datetime.min.replace(tzinfo=None) if ascending else datetime.max.replace(tzinfo=None)
            try:
                # Handle different date formats
                if 'D:' in str(date_str):  # PDF date format
                    date_str = date_str.replace('D:', '').split('+')[0].split('-')[0]
                    return datetime.strptime(date_str[:14], '%Y%m%d%H%M%S')
                
                # Parse ISO format and remove timezone info for comparison
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                # Remove timezone info to make it naive
                return dt.replace(tzinfo=None)
            except:
                return datetime.min.replace(tzinfo=None) if ascending else datetime.max.replace(tzinfo=None)
        
        return sorted(metadata_list, key=get_date, reverse=not ascending)


def run_metadata_analyzer():
    """Streamlit UI for metadata analyzer"""
    st.title("📄 Document Metadata Analyzer")
    st.caption("Extract and analyze metadata from documents, sort by date")
    
    st.divider()
    
    # File upload
    st.subheader("Upload Files")
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, XLSX, Images, etc.)",
        type=['pdf', 'docx', 'doc', 'xlsx', 'xls', 'csv', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp', 'pptx', 'txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        analyzer = DocumentMetadataAnalyzer()
        
        # Save uploaded files temporarily
        temp_dir = Path("temp_metadata_analysis")
        temp_dir.mkdir(exist_ok=True)
        
        file_paths = []
        for uploaded_file in uploaded_files:
            temp_path = temp_dir / uploaded_file.name
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(temp_path)
        
        # Analyze files
        with st.spinner("Analyzing metadata..."):
            metadata_list = analyzer.analyze_multiple_files(file_paths)
        
        # Sorting options
        st.subheader("Sort Options")
        col1, col2 = st.columns(2)
        
        with col1:
            sort_field = st.selectbox(
                "Sort by",
                ['modified', 'created', 'accessed', 'filename', 'size_bytes'],
                index=0
            )
        
        with col2:
            sort_order = st.radio(
                "Order",
                ['Newest First', 'Oldest First'],
                horizontal=True
            )
        
        ascending = sort_order == 'Oldest First'
        
        # Sort metadata
        if sort_field in ['modified', 'created', 'accessed']:
            sorted_metadata = analyzer.sort_by_date(metadata_list, sort_field, ascending)
        else:
            sorted_metadata = sorted(metadata_list, key=lambda x: x.get(sort_field, ''), reverse=not ascending)
        
        # Display results
        st.divider()
        st.subheader(f"Analysis Results ({len(sorted_metadata)} files)")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_size = sum(m.get('size_bytes', 0) for m in sorted_metadata)
            st.metric("Total Size", analyzer.format_file_size(total_size))
        with col2:
            file_types = len(set(m.get('type', 'Unknown') for m in sorted_metadata))
            st.metric("File Types", file_types)
        with col3:
            pdf_count = sum(1 for m in sorted_metadata if m.get('extension') == '.pdf')
            st.metric("PDF Files", pdf_count)
        with col4:
            doc_count = sum(1 for m in sorted_metadata if m.get('extension') in ['.docx', '.doc'])
            st.metric("Word Docs", doc_count)
        
        # Detailed view
        st.subheader("Detailed Metadata")
        
        for idx, metadata in enumerate(sorted_metadata, 1):
            # Format the title with date/time upfront with labels
            date_str = metadata.get('modified', 'N/A')
            filename = metadata.get('filename', 'Unknown')
            file_type = metadata.get('type', 'Unknown')
            
            # Parse and format with labels
            if date_str != 'N/A' and date_str is not None:
                try:
                    if 'D:' in str(date_str):  # PDF date format
                        date_str = date_str.replace('D:', '').split('+')[0].split('-')[0]
                        dt = datetime.strptime(date_str[:14], '%Y%m%d%H%M%S')
                    else:
                        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    
                    time_part = dt.strftime('%H:%M')
                    date_part = dt.strftime('%d %b %Y')
                    expander_title = f"⏰ Time: {time_part} | 📅 Date: {date_part} | 📄 {filename}"
                except:
                    expander_title = f"📄 {filename}"
            else:
                expander_title = f"📄 {filename}"
            
            with st.expander(expander_title):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**File Information**")
                    st.write(f"📁 Filename: `{metadata.get('filename', 'N/A')}`")
                    st.write(f"📊 Type: {metadata.get('type', 'N/A')}")
                    st.write(f"💾 Size: {metadata.get('size_readable', 'N/A')}")
                    st.write(f"📅 Created: {analyzer.format_datetime(metadata.get('created', 'N/A'))}")
                    st.write(f"✏️ Modified: {analyzer.format_datetime(metadata.get('modified', 'N/A'))}")
                
                with col2:
                    st.write("**Document Properties**")
                    if 'title' in metadata:
                        st.write(f"📝 Title: {metadata.get('title', 'N/A')}")
                    if 'author' in metadata:
                        st.write(f"👤 Author: {metadata.get('author', 'N/A')}")
                    if 'pages' in metadata:
                        st.write(f"📄 Pages: {metadata.get('pages', 'N/A')}")
                    if 'sheets' in metadata:
                        st.write(f"📊 Sheets: {metadata.get('sheets', 'N/A')}")
                    if 'size' in metadata and metadata.get('type') == 'Image':
                        st.write(f"🖼️ Dimensions: {metadata.get('size', 'N/A')}")
                
                # Download/Open file button
                st.divider()
                file_path = metadata.get('path')
                if file_path and os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                    
                    st.download_button(
                        label=f"📂 Open/Download {filename}",
                        data=file_data,
                        file_name=filename,
                        mime="application/octet-stream",
                        key=f"download_{idx}"
                    )
                
                # Show all metadata as JSON
                if st.checkbox(f"Show raw metadata", key=f"raw_{idx}"):
                    st.json(metadata)
        
        # Export option
        st.divider()
        if st.button("📥 Export Metadata to CSV"):
            df = pd.DataFrame(sorted_metadata)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"metadata_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Cleanup temp files
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
    
    else:
        st.info("👆 Upload files to analyze their metadata")
        
        st.markdown("### Supported Formats")
        st.markdown("- **Documents:** PDF, DOCX, DOC, TXT")
        st.markdown("- **Spreadsheets:** XLSX, XLS, CSV")
        st.markdown("- **Presentations:** PPTX, PPT")
        st.markdown("- **Images:** JPG, PNG, GIF, BMP, TIFF, WEBP")
        st.markdown("- **And more...**")
