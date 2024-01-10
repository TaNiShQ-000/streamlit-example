# import altair as alt
# import numpy as np
# import pandas as pd

import torch as tc
from transformers import pipeline

device = "cuda:0" if tc.cuda.is_available() else "cpu"
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

# import streamlit as 
