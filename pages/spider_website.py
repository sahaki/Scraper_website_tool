import streamlit as st
from dotenv import load_dotenv
import requests
import os
import pandas as pd

load_dotenv()

# Custom CSS to hide sidebar navigation
st.markdown("""
    <style>
        .stMainBlockContainer{width: 100%; max-width: 100%; padding: 6rem 3rem 10rem;}
    </style>
""", unsafe_allow_html=True)

# Page Title
st.title("🛠 Spider Website Tool")

# Description
st.write("This tool helps you spider link data from websites and return the urls list.")

# Initialize session state
if 'spider_results' not in st.session_state:
    st.session_state.spider_results = None

# List of unsupported file extensions
unsupported_extensions = (".jpg", ".jpeg", ".png", ".gif", ".zip", ".rar", ".exe", ".mp3", ".mp4", ".avi", ".mov")

# Form
with st.form(key="scraping_form"):
    # URL Input
    url = st.text_input("Enter URL:", placeholder="https://www.example.com")

    # Deep Level Input
    max_depth = st.selectbox("Max Depth", [1, 2, 3], index=1)

    # Submit Button
    submit_button = st.form_submit_button(label="Start Spidering")

# Process Form Submission
if submit_button:
    if url:
        # Check if URL contains an unsupported file extension
        if url.lower().endswith(unsupported_extensions):
            st.warning("The provided URL points to an unsupported file type (e.g., image, document, or compressed file). Please enter a general webpage URL or an XML link.")
        else:
            with st.spinner("Spidering in progress..."):
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/spider",
                        json={
                            "url": url,
                            "max_depth": max_depth
                        }
                    )
                    if response.status_code == 200:
                        # Store data in session state
                        st.session_state.spider_results = response.json()
                        st.success("Spidering completed successfully!")
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a valid URL before submitting.")

# Display results (this runs every time, not just on button click)
if st.session_state.spider_results:
    data = st.session_state.spider_results
    
    # Display summary information
    st.subheader("Spider Results")
    
    # Add clear results button
    if st.button("Clear Results"):
        st.session_state.spider_results = None
        st.rerun()
    
    st.write(f"**URL:** {data['url']}")
    st.write(f"**Number of links found:** {data['link_count']}")
    
     # Add warning about potential invalid URLs
    st.warning("Warning: Some of the searched urls may contain invalid urls. Please check all urls list again.")
    
    # Display the links in an expandable section
    with st.expander("View all links", expanded=False):
        # Create a numbered list of links
        for i, link in enumerate(data['links'], 1):
            st.write(f"{i}. [{link}]({link})")
    
    # Add option to download links as CSV
    if data['link_count'] > 0:
        links_df = pd.DataFrame(data['links'], columns=["URL"])
        csv = links_df.to_csv(index=False)
        
        st.download_button(
            label="Download links as CSV",
            data=csv,
            file_name="spider_results.csv",
            mime="text/csv"
        )