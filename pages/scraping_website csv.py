import streamlit as st
from dotenv import load_dotenv
import requests
import os

load_dotenv()

# Custom CSS to hide sidebar navigation
st.markdown("""
    <style>
        .stMainBlockContainer{width: 100%; max-width: 100%; padding: 6rem 3rem 10rem;}
    </style>
""", unsafe_allow_html=True)

# Page Title
st.title("🛠 Scraping Website Tool")

# Description
st.write("This tool helps you scrape data from websites and store it in Supabase.")

# List of unsupported file extensions
supported_extensions = (".csv", ".xls", ".xlsx")

# Form
with st.form(key="scraping_form"):
    # URL Input - Changed to file uploader
    uploaded_file = st.file_uploader("Upload a CSV or Excel file with URLs:", type=["csv", "xlsx", "xls"])

    # New Radio Option for Expand the details
    expand_details = st.radio(
        "Expand the details",
        ("No", "Yes"),
        index=0,  # default is "No"
        help="AI will help you generate more details of your content and generate Q&A content for the content in your URL."
    )

    # Readonly Source Name
    source_name = st.text_input("Define Source Name:", os.getenv('SUPABASE_SOURCE_TEXT'), disabled=True)

    # Submit Button
    submit_button = st.form_submit_button(label="Start Scraping")

# Process Form Submission
if submit_button:
    if uploaded_file is not None:
        # Get file extension
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        # Check if file has a supported extension
        if file_extension not in supported_extensions:
            st.warning(f"Invalid file format. Please upload a file with {', '.join(supported_extensions)} extension.")
        else:
            # Process the uploaded file
            try:
                # Read the file content
                if file_extension == '.csv':
                    import pandas as pd
                    df = pd.read_csv(uploaded_file)
                else:  # Excel file
                    import pandas as pd
                    df = pd.read_excel(uploaded_file)
                
                # Extract URLs from the first column
                if len(df.columns) > 0:
                    urls = df.iloc[:, 0].dropna().tolist()
                    url_str = ','.join(urls)  # Join URLs with comma for the API

                    st.success(f"URLs list is {url_str}")
                    st.stop()  # Stop further execution of the Streamlit script
                    
                    with st.spinner("Scraping in progress..."):
                        try:
                            response = requests.post(
                                "http://127.0.0.1:8000/crawl",
                                json={
                                    "scrape_type": 'Link',
                                    "url": url_str,
                                    "supabase_table": os.getenv("SUPABASE_TABLES"),
                                    "source_name": source_name,
                                    "expand_details": expand_details
                                }
                            )
                            if response.status_code == 200:
                                st.success(f"Scraping started successfully for {len(urls)} URLs!")
                            else:
                                st.error(f"Error: {response.status_code} - {response.text}")
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                else:
                    st.error("The uploaded file doesn't contain any columns.")
            except Exception as e:
                st.error(f"Error processing the file: {e}")
    else:
        st.error("Please upload a file before submitting.")