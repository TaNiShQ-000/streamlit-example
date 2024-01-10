import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
# import torch
from transformers import pipeline

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    "automatic-speech-recognition", model="openai/whisper-large-v3", device=device
)

st.write("What is data science ? ")

audio_file = st.file_uploader("student_recording", type=["wav"])


if audio_file is not None:
    # Read the content only if a file is uploaded
    audio_content = audio_file.read()
    
    # Process audio using your module
    result = pipe(audio_content, max_new_tokens=256)
    
    st.write("Processing Result:", result)


# st.altair_chart(alt.Chart(df, height=700, width=700)
#     .mark_point(filled=True)
#     .encode(
#         x=alt.X("x", axis=None),
#         y=alt.Y("y", axis=None),
#         color=alt.Color("idx", legend=None, scale=alt.Scale()),
#         size=alt.Size("rand", legend=None, scale=alt.Scale(range=[1, 150])),
#     ))
