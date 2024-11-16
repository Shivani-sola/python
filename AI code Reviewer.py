import streamlit as st
import google.generativeai as genai
import time

# Configure the API key for Google Generative AI
genai.configure(api_key="AIzaSyBuDij3utstYSsOOkDj860i0n_ZLuPNshI")  # Replace with your API key

# Set up the system prompt for code review
sys_prompt = '''You are a code review assistant. You read code, give bug reports, and fix any mistakes.
                It should be user-friendly, efficient, and provide accurate bug reports and fixed code snippets.
                You only respond when the user gives you code. If the user says anything else, simply say 'Sorry.'
                If the user asks to write code, you are not supposed to fulfill such queries. You can fix small code issues by assuming sometimes.
                If any lines are missing, add them so the code works.
                These are the following outputs you will provide:
                1. Bug report
                2. 👩‍💻Fixed code snippet
                3. Explanation and suggestions'''

# Initialize the generative model
model = genai.GenerativeModel("models/gemini-1.5-flash", system_instruction=sys_prompt)

# Function to generate the code review
def generate_code_review(ex_code):
    if ex_code.strip():  # Check if user entered code
        try:
            # Start time for execution time tracking
            start_time = time.time()

            st.markdown("### Generating code review...")
            time.sleep(1)  # Simulate loading time

            # Requesting the model to generate content based on the input code
            response = model.generate_content(ex_code)
            
            # End time for execution time tracking
            end_time = time.time()
            execution_time = end_time - start_time

            # Display the code review
            st.markdown("### Code Review Report:")
            st.markdown(response.text)  # Displaying the model's feedback

            # Additional suggestions can be added here
            st.markdown("### Suggestions for Improvement:")
            st.markdown("1. Avoid using `print` statements for debugging. Use logging instead.")
            st.markdown("2. Consider adding type hints for better readability.")

            # Display the original code
            st.markdown("### Original Code:")
            st.code(ex_code, language="python")
            
            # Display the code execution time
            st.markdown(f"### Code Execution Time: {execution_time:.4f} seconds")
        
        except Exception as e:
            st.markdown(f"<p style='color:red;'>Error: {e}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:red;'>Please enter some code to review.</p>", unsafe_allow_html=True)

# Streamlit input for the code
st.title("🤖BugFree: AI Code Reviewer")
code_input = st.text_area("Enter your code:", height=200)

# Streamlit button to trigger code review
if st.button("Generate Code Review"):
    generate_code_review(code_input)
