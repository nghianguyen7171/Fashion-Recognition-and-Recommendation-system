import numpy as np
import matplotlib.pyplot as plt
import cv2

from cloth_detection import Detect_Clothes_and_Crop
from utils_my import Read_Img_2_Tensor, Save_Image, Load_DeepFashion2_Yolov3

import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from PIL import  Image
import time
from plotly.offline import iplot

def app():
    header = st.container()
    add_img = st.container()


    
    with header:
        st.write('Recognize and extract fashion products in an image')
        #main_logo_path = 'logo/logo_cnu.png'
        #rrs_logo_path = 'image/logo_2.png'
        #main_logo = Image.open(main_logo_path).resize((200, 200))

    with add_img:
        st.write("Select your image")
        uploaded_file = st.file_uploader("Choose an image...", key='1')

        if uploaded_file is not None:
            model = Load_DeepFashion2_Yolov3()
            upload_path = os.path.join('images', uploaded_file.name)
            # read image
            img = Image.open(uploaded_file)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img_tensor = Read_Img_2_Tensor(upload_path)

            # Clothes detection and crop the image
            img_crop = Detect_Clothes_and_Crop(img_tensor, model)

            # Transform the image to gray_scale
            cloth_img = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)

            # Pretrained classifer parameters
            PEAK_COUNT_THRESHOLD = 0.02
            PEAK_VALUE_THRESHOLD = 0.01

            # show the image
            if st.button('Show uploaded image', key='upload'):
                st.image(img, caption='Uploaded Image.', use_column_width=True)
            

            if st.button('Classify the image', key='clf'):
                
                # Horizontal bins
                horizontal_bin = np.mean(cloth_img, axis=1)
                horizontal_bin_diff = horizontal_bin[1:] - horizontal_bin[0:-1]
                peak_count = len(horizontal_bin_diff[horizontal_bin_diff>PEAK_VALUE_THRESHOLD])/len(horizontal_bin_diff)
                if peak_count >= PEAK_COUNT_THRESHOLD:
                    st.write('Class 1 (clothes wtih stripes)')
                else:
                    st.write('Class 0 (clothes without stripes)')

                fig = plt.figure(figsize=(10, 10))
                rows = 2
                columns = 2
                fig.add_subplot(rows, columns, 1)
                
                plt.imshow(img)
                plt.axis('off')
                plt.title('Input image')
                #plt.show()

                fig.add_subplot(rows, columns, 2)
                plt.imshow(img_crop)
                plt.axis('off')
                plt.title('Cloth detection and crop')
                #plt.show()

                # show fig
                st.pyplot(fig)

            if st.button('Save result image', key='save'):
                with st.spinner('Saving image...'):
                    time.sleep(2)
                    Save_Image(img_crop, './images/test1_crop.jpg')
                st.write('Image saved successfully!')



