import streamlit as st
import pandas as pd
from PIL import Image
import base64
import matplotlib.pyplot as plt
import Recognition
import Detection
import Recommendation

# PAGES = {
#     "Fashion classification": Recognition,
#     "Fashion detection": Detection,
#     "Recommendation": Recommendation
# }


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('logo/bg.jpg')  

#main_logo_path = 'logo/logo_cnu.png'
#rrs_logo_path = 'image/logo_2.png'
#main_logo = Image.open(main_logo_path).resize((200, 200))

#st.image(main_logo)

font_css = """
<style>
button[data-baseweb="tab"] {
  font-size: 16px;
}
</style>
"""
st.write(font_css, unsafe_allow_html=True)

st.title('Fashion Recognition & Recommendation System')


tabs = st.tabs(["Classification", "Detection", "Recommendation"])
with tabs[0]:
    Recognition.app()

with tabs[1]:
    Detection.app()

with tabs[2]:
    Recommendation.app()

#selection = st.tabs.radio("Go to", list(PAGES.keys()))
#page = PAGES[selection]
#page.app()