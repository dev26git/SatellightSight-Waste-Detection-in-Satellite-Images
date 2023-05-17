from PIL import Image
import streamlit as st
import cv2
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

    st.write("##")
    st.subheader('An automated machine learning approach to detect waste dumps in satellite images')
    st.write("It uses texture analysis techniques like GLCM and LBP.")
    st.write("A state-of-the-art Sliding Window Algorithm to analyses sub-images. It is adapted to the training data used for the model.")
    st.write("A binary classifier model is used to classify each sub-image of a given satellite image into 2 categories- waste or non-waste.")
    # DISPLAY SAMPLE ANNOTATION
    # with st.container():
    #     original_img, arrow, annotated_img = st.columns((5, 1, 5))
    #     with original_img:
    #         st.image(Image.open("image_repo/sample_original_image.jpg"))
    #     with arrow:
    #         st.image(Image.open("image_repo/NavyBlueArrow.png"))
    #     with annotated_img:
    #         st.image(Image.open("image_repo/sample_annotated_image.jpg"))

    st.write("##")
    st.write("##")

    sample_images_folder = "image_repo/Sample_Satellite_Images"
    img_rgb = image_select(
        label="Select a Satellite Image",
        images=[
            cv2.cvtColor(cv2.imread(os.path.join(sample_images_folder, path)), cv2.COLOR_BGR2RGB) for path in os.listdir(sample_images_folder)
        ]
        # captions=os.listdir(sample_images_folder)
    )

    st.write("##")
    st.write("##")

    # MAIN CODE
    # if file is not None:
    if img_rgb is not None:

        with st.container():
            st.subheader("Model Output")
            st.write("Please wait for the model to work its magic.")
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
