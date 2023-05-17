from PIL import Image
import streamlit as st
import cv2
import numpy as np
import Prediction
from streamlit_image_select import image_select
import os


def main():
    st.set_page_config(page_title='SatellightSight', layout="wide")

    with st.container():
        _, middle, _ = st.columns((5, 1, 5))
        with middle:
            st.image("image_repo/Logo_of_Symbiosis_International_University.svg.png")
    with st.container():
        _, middle, _ = st.columns((4, 6, 1))
        with middle:
            st.subheader("Symbiosis Institute of Technology")

    st.write("##")
    st.write("##")
    st.write("##")

    # HEADER SECTION
    st.title(":blue[SatellightSight] :satellite:")
    # st.markdown("<h1 style='text-align: center; color: blue;'>SatellightSight</h1>", unsafe_allow_html=True)


    st.write("##")
    st.subheader('An automated machine learning approach to detect waste dumps in satellite images')
    st.write("##")

    # DISPLAY SAMPLE ANNOTATION
    with st.container():
        original_img, arrow, annotated_img = st.columns((5, 1, 5))
        with original_img:
            st.image(Image.open("image_repo/sample_original_image.jpg"))
        with arrow:
            st.image(Image.open("image_repo/NavyBlueArrow.png"))
        with annotated_img:
            st.image(Image.open("image_repo/sample_annotated_image.jpg"))


    st.write("##")
    st.write("##")
    st.write("##")
    st.write("##")


    # FILE UPLOAD
    # with st.container():
    #     file = st.file_uploader("**Upload a satellite image here**", type=['jpeg', 'jpg', 'png'])







    img_bgr = image_select(
        label="Select a Satellite Image",
        images=[
            cv2.imread(path) for path in os.listdir("image_repo")
        ]
        # ,captions=["Sample Image 1"],
    )















    st.write("##")
    st.write("##")


    # MAIN CODE
    # if file is not None:
    if img_bgr is not None:
        # read image
        # file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        # img_bgr = cv2.imdecode(file_bytes, 1)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with st.container():
            st.subheader("Model Output")
            user_input, arrow, output_image = st.columns((5, 1, 5))
            # Print input image
            with user_input:
                st.image(img_rgb, use_column_width=True)
            with arrow:
                st.image(Image.open("image_repo/NavyBlueArrow.png"))

            annotated_output_image = Prediction.getAnnotatedImage(img_rgb)

            # print annotated image
            with output_image:
                st.image(annotated_output_image)


if __name__ == "__main__":
    main()
