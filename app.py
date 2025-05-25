import cv2
import numpy as np
import re
import pandas as pd
import openai
import streamlit as st
import os
from openai import OpenAI
from openai import OpenAIError

# Set your OpenAI API key here or use environment variable
openai.api_key ='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'
import cv2
import pytesseract

# Point pytesseract to your Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load image in grayscale
image = cv2.imread('report.jpg', cv2.IMREAD_GRAYSCALE)

# Run OCR
text = pytesseract.image_to_string(image)

print(text)


def preprocess_image(image_path):
    image = cv2.imread(image_path, 0)  # grayscale
    denoised = cv2.fastNlMeansDenoising(image, None, 30, 7, 21)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def extract_text(image):
    # Using pytesseract for OCR
    import pytesseract
    text = pytesseract.image_to_string(image)
    return text

def parse_test_results(text):
    pattern = r'(?P<test>\w[\w\s]+):?\s+(?P<value>\d+(\.\d+)?)[\s]*?(?P<unit>[\w/]+)?[\s\-–]*?(?P<range>\d+[\-–]\d+)'
    matches = re.findall(pattern, text)
    
    data = []
    for m in matches:
        test, value, _, unit, normal_range = m
        low, high = map(float, re.split(r'[\-–]', normal_range))
        value = float(value)
        status = "Normal"
        if value < low:
            status = "Low"
        elif value > high:
            status = "High"
        data.append([test.strip(), value, unit, normal_range, status])
    
    df = pd.DataFrame(data, columns=['Test', 'Value', 'Unit', 'Normal Range', 'Status'])
    return df

import openai

openai.api_key = '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'

def explain_result(test, value, normal_range):
    prompt = f"Explain in simple terms what it means if a patient's {test} is {value} given that the normal range is {normal_range}."
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except openai.RateLimitError:
        return "API rate limit exceeded. Please try again later."
    except openai.AuthenticationError:
        return "Authentication error. Please check your API key."
    except openai.APIConnectionError:
        return "Network error. Please check your internet connection."
    except openai.OpenAIError as e:
        return f"OpenAI API error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"

openai.api_key = st.secrets["openai_api_key"]



def generate_summary(df):
    summary = []
    abnormal_df = df[df['Status'] != 'Normal']
    st.write("🔍 Abnormal values being explained:", abnormal_df)

    for _, row in abnormal_df.iterrows():
        explanation = explain_result(row['Test'], row['Value'], row['Normal Range'])
        st.write(f"🧠 Explanation for {row['Test']} = {row['Value']} (Range: {row['Normal Range']}):", explanation)
        summary.append(f"{row['Test']}: {explanation}")
    
    if not summary:
        return "✅ All test values are within the normal range."
    return "\n\n".join(summary)


# Streamlit UI
st.title("🧪 AI Medical Report Analyzer")

uploaded_file = st.file_uploader("Upload a scanned medical report (PNG/JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.read())

    image = preprocess_image("temp_image.png")
    text = extract_text(image)
    st.text_area("Extracted Text", text, height=300)

    df = parse_test_results(text)
    st.dataframe(df)

    if st.button("Generate Explanations"):
        df['Explanation'] = df.apply(
            lambda row: explain_result(row['Test'], row['Value'], row['Normal Range']) if row['Status'] != 'Normal' else "", axis=1
        )
        for _, row in df[df['Explanation'] != ""].iterrows():
            with st.expander(f"{row['Test']} ({row['Status']})"):
                st.write(row['Explanation'])

    if st.button("Generate Summary"):
        summary = generate_summary(df)
        st.text_area("Summary", summary, height=200)

