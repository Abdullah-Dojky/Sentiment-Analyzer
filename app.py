import streamlit as st
from transformers import pipeline

@st.cache_resource
def load():
    return pipeline("sentiment-analysis")

st.title("Sentiment Analyzer")

txt = st.text_area("Your text:")

if st.button("Analyze") and txt.strip():
    classifier = load()
    r = classifier(txt)[0]
    st.subheader(r["label"])
    st.write(f"Confidence: {r['score']:.1%}")