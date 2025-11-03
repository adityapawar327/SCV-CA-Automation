import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from difflib import SequenceMatcher
import io
import tempfile
import os
import easyocr
import fitz  # PyMuPDF
import cv2

# Initialize EasyOCR reader (cached to avoid reloading)
@st.cache_resource
def get_ocr_reader():
    """Initialize and cache EasyOCR reader optimized for English text"""
    return easyocr.Reader(['en'], gpu=False, model_storage_directory='./models')

def preprocess_image_for_ocr(image):
    """Preprocess image for better OCR accuracy"""
    import cv2
    
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Apply adaptive thresholding for better text detection
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    
    return denoised

def extract_text_with_boxes(image):
    """Extract text and bounding boxes from image using EasyOCR optimized for English text"""
    reader = get_ocr_reader()
    
    # Preprocess image for better accuracy
    processed_img = preprocess_image_for_ocr(image)
    
    # Perform OCR with optimized parameters for text documents
    results = reader.readtext(
        processed_img,
        detail=1,
        paragraph=False,
        min_size=10,
        text_threshold=0.7,
        low_text=0.4,
        link_threshold=0.4,
        canvas_size=2560,
        mag_ratio=1.5,
        slope_ths=0.1,
        ycenter_ths=0.5,
        height_ths=0.5,
        width_ths=0.5,
        add_margin=0.1
    )
    
    text_blocks = []
    for (bbox, text, conf) in results:
        # Filter out low confidence detections
        if conf < 0.3:
            continue
            
        # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        left = int(min(x_coords))
        top = int(min(y_coords))
        width = int(max(x_coords) - left)
        height = int(max(y_coords) - top)
        
        # Clean up text - remove extra spaces and normalize
        cleaned_text = ' '.join(text.split())
        
        text_blocks.append({
            'text': cleaned_text,
            'left': left,
            'top': top,
            'width': width,
            'height': height,
            'conf': conf * 100  # Convert to percentage
        })
    
    return text_blocks

def pdf_to_images(pdf_file):
    """Convert PDF pages to images using PyMuPDF with high quality for text"""
    # Read PDF bytes
    pdf_bytes = pdf_file.read()
    
    # Open PDF from bytes
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    images = []
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        
        # Render page to image at 400 DPI for better text clarity
        mat = fitz.Matrix(400/72, 400/72)  # 400 DPI scaling for better OCR
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    
    pdf_document.close()
    return images

def normalize_text(text):
    """Normalize text for better comparison"""
    import re
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Normalize common Indian English variations
    text = text.replace('Rs.', 'Rs')
    text = text.replace('₹', 'Rs')
    # Remove punctuation variations
    text = re.sub(r'[,\s]+', ' ', text)
    return text.strip()

def compare_text_blocks(blocks1, blocks2):
    """Compare text blocks and find differences with improved accuracy"""
    # Create normalized text for comparison
    text1 = ' '.join([normalize_text(b['text']) for b in blocks1])
    text2 = ' '.join([normalize_text(b['text']) for b in blocks2])
    
    # Use SequenceMatcher with autojunk disabled for better accuracy
    matcher = SequenceMatcher(None, text1, text2, autojunk=False)
    opcodes = matcher.get_opcodes()
    
    diff_blocks1 = []
    diff_blocks2 = []
    diff_blocks1_set = set()
    diff_blocks2_set = set()
    
    # Map character positions to blocks with better accuracy
    for tag, i1, i2, j1, j2 in opcodes:
        if tag in ['replace', 'delete']:
            # Find blocks in image1 that contain differences
            char_pos = 0
            for idx, block in enumerate(blocks1):
                block_text = normalize_text(block['text'])
                block_len = len(block_text) + 1
                block_start = char_pos
                block_end = char_pos + block_len
                
                # Check if this block overlaps with the difference range
                if (block_start <= i1 < block_end) or (block_start < i2 <= block_end) or (i1 <= block_start and block_end <= i2):
                    if idx not in diff_blocks1_set:
                        diff_blocks1.append(block)
                        diff_blocks1_set.add(idx)
                
                char_pos += block_len
        
        if tag in ['replace', 'insert']:
            # Find blocks in image2 that contain differences
            char_pos = 0
            for idx, block in enumerate(blocks2):
                block_text = normalize_text(block['text'])
                block_len = len(block_text) + 1
                block_start = char_pos
                block_end = char_pos + block_len
                
                # Check if this block overlaps with the difference range
                if (block_start <= j1 < block_end) or (block_start < j2 <= block_end) or (j1 <= block_start and block_end <= j2):
                    if idx not in diff_blocks2_set:
                        diff_blocks2.append(block)
                        diff_blocks2_set.add(idx)
                
                char_pos += block_len
    
    return diff_blocks1, diff_blocks2

def annotate_image(image, diff_blocks, color='black'):
    """Draw boxes on image for differences"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    for block in diff_blocks:
        x = block['left']
        y = block['top']
        w = block['width']
        h = block['height']
        
        # Draw black outline only
        if color == 'black':
            outline_color = (0, 0, 0, 255)
        elif color == 'red':
            outline_color = (255, 0, 0, 255)
        elif color == 'green':
            outline_color = (0, 255, 0, 255)
        else:
            outline_color = (0, 0, 0, 255)
        
        draw.rectangle([x, y, x+w, y+h], outline=outline_color, width=3)
    
    return img_copy

def run_ocr_visual_diff():
    st.title("🔍 PDF & Image Comparator")
    st.caption("Compare images and PDFs with visual annotations (powered by EasyOCR)")
    
    # Initialize session state
    if 'ocr_files_data' not in st.session_state:
        st.session_state.ocr_files_data = {}
    if 'ocr_comparison_done' not in st.session_state:
        st.session_state.ocr_comparison_done = False
    
    st.divider()
    
    uploaded_files = st.file_uploader(
        "Upload image/PDF files (up to 15)",
        type=['png', 'jpg', 'jpeg', 'pdf'],
        accept_multiple_files=True,
        help="Upload multiple files to compare them pairwise"
    )
    
    if uploaded_files:
        if len(uploaded_files) > 15:
            st.warning("Maximum 15 files allowed. Processing first 15 files.")
            uploaded_files = uploaded_files[:15]
        
        st.info(f"Loaded {len(uploaded_files)} file(s)")
        
        # Load all files
        file_names = [f.name for f in uploaded_files]
        if set(file_names) != set(st.session_state.ocr_files_data.keys()):
            with st.spinner("Loading files..."):
                ocr_files_data = {}
                for file in uploaded_files:
                    if file.name.lower().endswith('.pdf'):
                        images = pdf_to_images(file)
                    else:
                        images = [Image.open(file)]
                    ocr_files_data[file.name] = images
                
                st.session_state.ocr_files_data = ocr_files_data
                st.session_state.ocr_comparison_done = False
        
        if len(st.session_state.ocr_files_data) < 2:
            st.error("Need at least 2 valid files to compare")
            return
        
        # File selection for comparison
        st.markdown("### Select Files to Compare")
        available_files = list(st.session_state.ocr_files_data.keys())
        
        col1, col2 = st.columns(2)
        with col1:
            file1 = st.selectbox("Base File", available_files, key='base_file_selector')
        with col2:
            compare_files = [f for f in available_files if f != file1]
            selected_compare_files = st.multiselect(
                "Compare Against",
                compare_files,
                default=compare_files[:1] if compare_files else [],
                key='compare_files_selector'
            )
        
        if not selected_compare_files:
            st.warning("Please select at least one file to compare against")
            return
        
        if st.button("🔍 Compare Files", type="primary", use_container_width=True):
            st.session_state.ocr_comparison_done = True
            st.session_state.selected_base_file = file1
            st.session_state.selected_compare_files = selected_compare_files
        
        if st.session_state.ocr_comparison_done:
            base_file = st.session_state.selected_base_file
            compare_files = st.session_state.selected_compare_files
            
            base_images = st.session_state.ocr_files_data[base_file]
            
            # Create tabs for each comparison
            tabs = st.tabs(compare_files + ["Summary"])
            
            all_differences = []
            
            for idx, compare_file in enumerate(compare_files):
                with tabs[idx]:
                    st.markdown(f"### {base_file} vs {compare_file}")
                    
                    compare_images = st.session_state.ocr_files_data[compare_file]
                    max_pages = min(len(base_images), len(compare_images))
                    
                    st.info(f"Base: {len(base_images)} page(s) | Compare: {len(compare_images)} page(s)")
                    
                    # Page selection
                    if max_pages > 1:
                        page_num = st.slider(
                            "Select page to compare",
                            1, max_pages, 1,
                            key=f"page_slider_{compare_file}"
                        ) - 1
                    else:
                        page_num = 0
                        st.caption("Comparing page 1")
                    
                    with st.spinner("Performing OCR and comparison..."):
                        img1 = base_images[page_num]
                        img2 = compare_images[page_num]
                        
                        # Extract text with bounding boxes
                        blocks1 = extract_text_with_boxes(img1)
                        blocks2 = extract_text_with_boxes(img2)
                        
                        # Compare and find differences
                        diff_blocks1, diff_blocks2 = compare_text_blocks(blocks1, blocks2)
                        

                        
                        # Display results - both images side by side
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**{base_file} (Original)**")
                            st.image(img1, width=None)
                        
                        with col2:
                            st.markdown(f"**{compare_file} (Differences Highlighted)**")
                            
                            # Annotate only the compare image with black outlines
                            annotated_compare = annotate_image(img2, diff_blocks2, color='black')
                            
                            st.image(annotated_compare, width=None)
                            st.caption(f"Found {len(diff_blocks2)} difference(s)")
                            
                            # Download button
                            buf = io.BytesIO()
                            annotated_compare.save(buf, format='PNG')
                            st.download_button(
                                "⬇️ Download Annotated Image",
                                buf.getvalue(),
                                f"annotated_{compare_file}_page{page_num+1}.png",
                                "image/png",
                                key=f"dl_{compare_file}_{page_num}",
                                use_container_width=True
                            )
                        
                        # Text differences
                        with st.expander("📝 View Text Differences"):
                            text1 = ' '.join([b['text'] for b in blocks1])
                            text2 = ' '.join([b['text'] for b in blocks2])
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.text_area(
                                    f"Text from {base_file}",
                                    text1,
                                    height=200,
                                    key=f"text1_{compare_file}_{page_num}"
                                )
                            with col2:
                                st.text_area(
                                    f"Text from {compare_file}",
                                    text2,
                                    height=200,
                                    key=f"text2_{compare_file}_{page_num}"
                                )
                        
                        # Store differences for summary
                        all_differences.append({
                            'Base File': base_file,
                            'Compare File': compare_file,
                            'Page': page_num + 1,
                            'Differences in Base': len(diff_blocks1),
                            'Differences in Compare': len(diff_blocks2),
                            'Total Differences': len(diff_blocks1) + len(diff_blocks2)
                        })
            
            # Summary tab
            with tabs[-1]:
                st.markdown("### Comparison Summary")
                
                if all_differences:
                    import pandas as pd
                    summary_df = pd.DataFrame(all_differences)
                    
                    st.markdown(f"**Total Comparisons: {len(all_differences)}**")
                    st.dataframe(summary_df)
                    
                    # Export summary
                    csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Summary (CSV)",
                        data=csv,
                        file_name="ocr_comparison_summary.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Statistics
                    st.divider()
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Differences", summary_df['Total Differences'].sum())
                    with col2:
                        st.metric("Avg Differences per Page", f"{summary_df['Total Differences'].mean():.1f}")
                    with col3:
                        st.metric("Max Differences", summary_df['Total Differences'].max())
                else:
                    st.success("No differences found")
    
    else:
        st.info("👆 Upload files to begin comparison")
        
        with st.expander("ℹ️ How to use"):
            st.markdown("""
            1. Upload multiple images or PDF files (up to 15)
            2. Select a base file and files to compare against
            3. Click 'Compare Files' to see visual differences
            4. Red boxes show deletions/changes in base file
            5. Green boxes show additions/changes in compare files
            6. View each comparison in separate tabs
            7. Download annotated images and summary report
            8. For multi-page PDFs, use the slider to compare different pages
            """)
