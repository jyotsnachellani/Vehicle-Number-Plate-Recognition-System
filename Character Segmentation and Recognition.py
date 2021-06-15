from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import streamlit as st


def extract_plate(img): # the function detects and performs blurring on the number plate.
    image = img.copy()
    labels = open('obj.names.txt').read().strip().split('\n')#reading pretrained object labels
    weights = 'yolov4-obj_last.weights'#pretrained weights
    architecture = 'yolov4-obj.cfg.txt'#Neural network architecture
    model = cv2.dnn.readNet(architecture,weights)#reading the neural network
    Height,Width = image.shape[:2]#for scaling our bounding box wrt to image
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (608, 608), swapRB=True, crop=False)
    blob.shape
    model.setInput(blob)
    def forward_prop(model):
        layers = model.getLayerNames()#get all layer names
        output_layer = [layers[i[0]-1] for i in model.getUnconnectedOutLayers()]#get names of output layers
        return model.forward(output_layer)#does forward prop and gives output for the layers passed to it
    CONFIDENCE = 0.5
    IOU_THRESHOLD = 0.5
    boxes, confidences, class_ids = [], [], []
    outputs = forward_prop(model)#outputs of the last layer of NN 
    for output in outputs:#loop through output of NN
        for pred in output:#loop through all predictions
            class_scores = pred[5:]#get class probs
            class_id = np.argmax(class_scores)#find max class prob
            class_prob = class_scores[class_id]#get class_id of class with max prob
            if class_prob>CONFIDENCE:
                #scale dimensions of bounding box wrt to image
                center_x = int(pred[0] * Width)
                center_y = int(pred[1] * Height)
                w = int(pred[2] * Width)
                h = int(pred[3] * Height)
                #getting top-left co-ordinates
                x = center_x - (w / 2)
                y = center_y - (h / 2)
                class_ids.append(class_id)
                confidences.append(float(class_prob))
                boxes.append([x, y, w, h])
    indices=cv2.dnn.NMSBoxes(boxes,confidences,CONFIDENCE,IOU_THRESHOLD)#performs nms and returns boxes to be kept
    def bounding_box(image,x,y,w,h,class_id,prob):
        label = labels[class_id]#getting label
        text = label+' : '+str(prob)
        cv2.rectangle(image,(x-3,y),(x+w+3,y+h),color=(255,0,0),thickness=2)
    #print(indices.flatten().max())
    for i in indices.flatten():
        i=indices.flatten().max()
        x,y,w,h = boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]
        prob=round(confidences[i], 3)
        bounding_box(image,int(x),int(y),int(w),int(h), class_ids[i] ,prob )
    print(x,y,x+w,y+h)
    #cv2.imshow("object detection", image)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    #cv2.imwrite("plate_1.jpg",image[int(y)+2:int(y)+int(h)-2,int(x)+2:int(x)+int(w)-2])
    return image[int(y)+2:int(y)+int(h)-2,int(x)+2:int(x)+int(w)-2]



def find_contours(plate) :

    # Find all contours in the image
    cntrs, _ = cv2.findContours(plate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    ii = cv2.imread('contour.jpg')
    
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        #detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        ratio= intHeight/intWidth
        #checking the dimensions of the contour to filter out the characters by contour's size
        if 1.5<=ratio<=2.8: # Only select contour with defined ratio
            if intHeight/plate.shape[0]>=0.333:
                x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

                char_copy = np.zeros((44,24))
                #extracting each character using the enclosing rectangle's coordinates.
                char = plate[intY:intY+intHeight, intX:intX+intWidth]
                char = cv2.resize(char, (20, 40))
            
                cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
                plt.imshow(ii, cmap='gray')

#             Make result formatted for classification: invert colors
                char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
                char_copy[2:42, 2:22] = char
                char_copy[0:2, :] = 0
                char_copy[:, 0:2] = 0
                char_copy[42:44, :] = 0
                char_copy[:, 22:24] = 0

                img_res.append(char_copy) #List that stores the character's binary image (unsorted)
            
    #Return characters on ascending order with respect to the x-coordinate
            
    #plt.show()
    #arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])# stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res


# Find characters in the resulting images
def segment_characters(image) :

    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

   
    plt.imshow(img_binary_lp, cmap='gray')
    #plt.show()
    cv2.imwrite('contour.jpg',img_binary_lp)
    # Get contours within cropped license plate
    char_list = find_contours(img_binary_lp)

    return char_list


def fix_dimension(img): 
    new_img = np.zeros((28,28,3))
    for i in range(3):
        new_img[:,:,i] = img
    return new_img
  
def show_results(char):
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c

    output = []
    for i,ch in enumerate(char): #iterating over the characters
        img_ = cv2.resize(ch, (28,28))
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3) #preparing image for the model
        y_ = model.predict_classes(img)[0] #predicting the class
        character = dic[y_] #
        output.append(character) #storing the result in a list
    plate_number = ''.join(output)
    return plate_number


st.set_option('deprecation.showfileUploaderEncoding',False)

@st.cache(allow_output_mutation=True)

def load_model():
    model = keras.models.load_model('number_plates_final.h5')
    return model
model = load_model()
st.markdown("""
<style>
.big-font {
    font-size:50px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Number Plate Detection </p>', unsafe_allow_html=True)
file = st.file_uploader("Please upload an image")
import cv2
from PIL import Image

def import_and_predict(image,model):
   
    image = np.array(image)
    char = segment_characters(extract_plate(image))
    x = show_results(char)
    return x
if file is None:
    #st.text("Please upload an image file")
    pass
else:
    image = Image.open(file)
    st.image(image,use_column_width=True)
    x = import_and_predict(image,model)
    st.markdown('<p style="font-size:20px;">Characters of Number Plate are: </p>', unsafe_allow_html=True)
    st.success(x)