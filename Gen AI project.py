import streamlit as st
from PIL import Image
import pytesseract
import pyttsx3
import google.generativeai as genai
import numpy as np
import cv2
import re
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import GoogleGenerativeAI

# Configure Google Generative AI API
API_KEY = "AIzaSyD22QE982Waq_JAAogNZVS7NwNR3MSL7AE"  # Replace with your actual API key
genai.configure(api_key=API_KEY)

# Functions
def real_time_scene_understanding(scene_image):
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash-001")
        response = model.generate_content(scene_image)
        return response.text
    except Exception as e:
        return f"Error in real-time scene understanding: {e}"

def text_to_speech(image):
    try:
        # Convert PIL Image to OpenCV format (NumPy array)
        image = np.array(image)
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Extract text using pytesseract
        text = pytesseract.image_to_string(gray)

        # Initialize TTS engine
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

        return text
    except Exception as e:
        return f"Error in text-to-speech conversion: {e}"

def parse_bounding_box(response):
    bounding_boxes = re.findall(r'\[(\d+,\s*\d+,\s*\d+,\s*\d+,\s*[\w\s]+)\]', response)
    parsed_boxes = []
    for box in bounding_boxes:
        parts = box.split(',')
        ymin, xmin, ymax, xmax = map(int, parts[:4])
        label = parts[4].strip()
        parsed_boxes.append(((ymin, xmin, ymax, xmax), label))
    return parsed_boxes

def draw_bounding_boxes(image, bounding_boxes_with_labels):
    label_colors = {}
    image = np.array(image)
    for bounding_box, label in bounding_boxes_with_labels:
        ymin, xmin, ymax, xmax = bounding_box
        if label not in label_colors:
            label_colors[label] = np.random.randint(0, 256, size=(3,)).tolist()
        color = tuple(map(int, label_colors[label]))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return Image.fromarray(image)

def object_detection(image):
    try:
        # Convert the image to an appropriate format
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        img = Image.open(image)
        response = model.generate_content([
        img,
        (
        "Return bounding boxes for all objects in the image in the following format as"
        " a list. \n [ymin, xmin, ymax, xmax, object_name]. If there are more than one object, return separate lists for each object"
        ),])

        # Extract the relevant text from the response object
        
        response_text = response.candidates[0].content.parts[0].text
        bounding_boxes = parse_bounding_box(response_text)
        return bounding_boxes
    except Exception as e:
        return f"Error in object detection: {e}"

def scene_analysis_with_langchain(image_description, user_question):
    try:
        template = """
        You are an intelligent assistant. Here is a description of an image:
        "{description}"
        Answer the following question based on the image:
        "{user_question}"
        """
        prompt = PromptTemplate(template=template, input_variables=["description", "user_question"])
        llm = GoogleGenerativeAI(google_api_key=API_KEY, model="gemini-pro")
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run({"description": image_description, "user_question": user_question})
        return response
    except Exception as e:
        return f"Error in scene analysis with LangChain: {e}"

# Streamlit App
st.title("AI-Powered Image Analysis Tool")

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Real-Time Scene Understanding
    if st.button("Analyze Scene"):
        st.write("Analyzing the scene...")
        description = real_time_scene_understanding(img)
        if description:
            st.success(f"Scene Description: {description}")
        else:
            st.error("Failed to analyze the scene.")

    # Text-to-Speech
    if st.button("Extract Text and Convert to Speech"):
        text = text_to_speech(img)
        if text:
            st.success(f"Extracted Text: {text}")
        else:
            st.error("No text could be extracted from the image.")

    # Object Detection
    if st.button("Perform Object Detection"):
        img_path = (r"image_astro.jpg")
        bounding_boxes = object_detection(img_path)
        if isinstance(bounding_boxes, str):  # Error message handling
            st.error(bounding_boxes)
        elif bounding_boxes:
           img = Image.open(img_path)
           output_image = draw_bounding_boxes(img, bounding_boxes)
           st.image(output_image, caption="Detected Objects", use_column_width=True)
        else:
            st.warning("No objects were detected.")

    # Question Answering
    user_question = st.text_input("Ask a question about the image")
    if st.button("Get Answer"):
        if user_question:
            description = real_time_scene_understanding(img)
            if description:
                answer = scene_analysis_with_langchain(description, user_question)
                st.success(f"Answer: {answer}")
            else:
                st.error("Failed to get scene description.")
        else:
            st.error("Please enter a question.")
