import streamlit as st

import traceback
from news_bias_detector import fetch_article_async, detect_bias
import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

st.set_page_config(page_title="News Bias Detector", layout="centered")
st.title("📰 Autonomous News Bias Detector")

url = st.text_input("Enter a news article URL:")
if url:
    with st.spinner("Analyzing article for bias..."):
        try:
            text = asyncio.run(fetch_article_async(url))
            result = detect_bias(text)
            st.subheader("Predicted Sentiment:")

            label_map = {
                "negative": "Left",
                "neutral": "Center",
                "positive": "Right"
            }

            if isinstance(result, list) and len(result) > 0:
                entry = result[0]
                readable = label_map.get(entry["label"], entry["label"])
                dic = {
                    "label": readable,
                    "confidence": round(entry["score"], 4)
                }
                st.json(dic)
            else:
                st.warning("Unexpected model output.")

        except Exception:
            st.error("An error occurred:")
            st.error(traceback.format_exc())
