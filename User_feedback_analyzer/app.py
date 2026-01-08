import streamlit as st
from conditional_chain import analyze_feedback

st.set_page_config(
    page_title="Feedback Analyzer",
    page_icon="📝",
    layout="centered"
)

st.title("📝 AI Feedback Analyzer")
st.write("Enter user feedback to analyze sentiment and generate a response.")

feedback = st.text_area(
    "Enter Feedback:",
    placeholder="This phone has amazing battery life..."
)

if st.button("Analyze Feedback"):
    if feedback.strip() == "":
        st.warning("Please enter some feedback.")
    else:
        with st.spinner("Analyzing..."):
            response = analyze_feedback(feedback)

        st.success("AI Response:")
        st.write(response)
