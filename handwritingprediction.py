import cv2
import numpy as np
import tensorflow as tf

state=False

m_new = tf.keras.models.load_model('model_digit.h5')

img = np.ones([400,400],dtype='uint8')*255

img[50:350,50:350]=0

wname = 'Canvas'
cv2.namedWindow(wname)  # Really important for the output

def shape(event,x,y,flags,param):
    global state
    #global state
    if event == cv2.EVENT_LBUTTONDOWN:
        state=True
        cv2.circle(img,(x,y),10,(255,255,255),-1)
        
        if event == cv2.EVENT_MOUSEMOVE:
            if (state == True):
                cv2.circle(img,(x,y),10,(255,255,255),-1)
                print(x,y)
    else:
            state=False
        
cv2.setMouseCallback(wname,shape)    # Window Name, Function to be defined for mouse event

while True:
    cv2.imshow(wname,img)
    key = cv2.waitKey(1) # Declaring a variable named key
    if key== ord('q'): #  q for Quit
        break
    elif key == ord('c'): # clearing the screen
        img[50:350,50:350]=0
    elif key == ord('p'): # Predicting the Digit
        image_test  = img[50:350,50:350]
        image_test = cv2.resize(image_test,(28,28)).reshape(1,28,28)
        result = m_new.predict_classes(image_test)
        print('Digit recognised is :',result)
cv2.destroyAllWindows()
