import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎭")


@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


st.caption(
    "Sentiment Analyzer · Abdullah Abdul Wahid (B04-0923-000007) & Harmain Ansar (B04-0923-000006) · "
    "Streamlit + Hugging Face model (distilbert-base-uncased-finetuned-sst-2-english). "
    "If the app is asleep, click \"Yes, get this app back up\" and wait about 30 seconds."
)

st.title("Sentiment Analyzer")
st.write("Type a sentence and get an instant POSITIVE / NEGATIVE verdict with a confidence score.")

txt = st.text_area("Your text:", height=120)

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

st.divider()
st.caption(
    "Model: distilbert-base-uncased-finetuned-sst-2-english (binary POSITIVE/NEGATIVE, trained on movie reviews). "
    "Long texts are truncated. Sarcasm and mixed feelings may be misread."
)
