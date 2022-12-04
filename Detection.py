
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from PIL import  Image
import time
from plotly.offline import iplot

import numpy as np
import time
import tensorflow as tf
import cv2

from utils_my import Read_Img_2_Tensor, Load_DeepFashion2_Yolov3, Draw_Bounding_Box

def app():
    header = st.container()
    add_img = st.container()

    
    with header:
        st.write('Detect fashion products in an image')
        #main_logo_path = 'logo/logo_cnu.png'
        #rrs_logo_path = 'image/logo_2.png'
        #main_logo = Image.open(main_logo_path).resize((200, 200))

    with add_img:
        model = Load_DeepFashion2_Yolov3()
        def Detect_Clothes(img, model_yolov3, eager_execution=True):
            """Detect clothes in an image using Yolo-v3 model trained on DeepFashion2 dataset"""
            img = tf.image.resize(img, (416, 416))

            t1 = time.time()
            if eager_execution==True:
                boxes, scores, classes, nums = model_yolov3(img)
                # change eager tensor to numpy array
                boxes, scores, classes, nums = boxes.numpy(), scores.numpy(), classes.numpy(), nums.numpy()
            else:
                boxes, scores, classes, nums = model_yolov3.predict(img)
            t2 = time.time()
            print('Yolo-v3 feed forward: {:.2f} sec'.format(t2 - t1))

            class_names = ['short_sleeve_top', 'long_sleeve_top', 'short_sleeve_outwear', 'long_sleeve_outwear',
                        'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeve_dress',
                        'long_sleeve_dress', 'vest_dress', 'sling_dress']

            # Parse tensor
            list_obj = []
            for i in range(nums[0]):
                obj = {'label':class_names[int(classes[0][i])], 'confidence':scores[0][i]}
                obj['x1'] = boxes[0][i][0]
                obj['y1'] = boxes[0][i][1]
                obj['x2'] = boxes[0][i][2]
                obj['y2'] = boxes[0][i][3]
                list_obj.append(obj)

            return list_obj

        def Detect_Clothes_and_Crop(img_tensor, model, threshold=0.5):
            list_obj = Detect_Clothes(img_tensor, model)

            img = np.squeeze(img_tensor.numpy())
            img_width = img.shape[1]
            img_height = img.shape[0]

            # crop out one cloth
            for obj in list_obj:
                if obj['label'] == 'short_sleeve_top' and obj['confidence']>threshold:
                    img_crop = img[int(obj['y1']*img_height):int(obj['y2']*img_height), int(obj['x1']*img_width):int(obj['x2']*img_width), :]

            return img_crop

        st.write("Select your image")
        uploaded_file = st.file_uploader("Choose an image...", key='image')

        if uploaded_file is not None:
                upload_path = os.path.join('images', uploaded_file.name)
                # read image
                img = Image.open(uploaded_file)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                img_tensor = Read_Img_2_Tensor(upload_path)

                list_obj = Detect_Clothes(img_tensor, model)
                img_with_boxes = Draw_Bounding_Box(img_tensor, list_obj)

                rs_img = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)

                # show the image
                if st.button('Show uploaded image', key='show'):
                    st.image(img, caption='Uploaded Image.', use_column_width=True)

                
                if st.button('Show detected results', key='detect'):
                    # show rs_img
                    fig = plt.figure(figsize=(10, 10))
                    rows = 2
                    columns = 2
                    fig.add_subplot(rows, columns, 1)
                    
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title('Input image')
                #plt.show()

                    fig.add_subplot(rows, columns, 2)
                    plt.imshow(rs_img)
                    plt.axis('off')
                    plt.title('Detected fashion product')
                    #plt.show()

                    # show fig
                    st.pyplot(fig)
                    st.write('Detected fashion product: {}'.format(list_obj))

            # cv2.imshow("Clothes detection", cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # cv2.imwrite("./images/test6_clothes_detected.jpg", cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)*255)

        