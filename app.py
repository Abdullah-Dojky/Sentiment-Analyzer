import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎭")


@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


st.title("Sentiment Analyzer — Abdullah Abdul Wahid (B04-0923-000007) & Harmain Ansar (B04-0923-000006)")
st.caption("Type a sentence and get an instant POSITIVE / NEGATIVE verdict with a confidence score.")

txt = st.text_area("Your text:", height=120, placeholder="e.g. I absolutely loved this movie!")

if st.button("Analyze"):
    if not txt.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        try:
            with st.spinner("Analyzing..."):
                classifier = load_model()
                r = classifier(txt)[0]

            label = r["label"]
            score = r["score"]

            if label == "POSITIVE":
                st.subheader(f":green[{label}]")
            else:
                st.subheader(f":red[{label}]")
            st.write(f"Confidence: {score:.1%}")
        except Exception as e:
            st.error(f"Something went wrong while analyzing your text: {e}")
