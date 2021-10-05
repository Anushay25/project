from tkinter import *
from tkinter import filedialog
from PIL import Image,ImageTk
import cv2
import numpy as np
import joblib
from skimage.feature import hog
import os
from scipy.interpolate import interp1d
from tkinter import messagebox as mb

global showid,grimage,gthreshold,gstring,threshold_level,image_mode,filename
image_mode = False
threshold_level = 95

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,600)

clf, pp = joblib.load("digits_cls.pkl")
window = Tk()
window.title("Hand Written digit recognition")
window.geometry('1010x650')
window.resizable(0,0)

top_frame = Frame(window)
left_frame = Label(top_frame,width=500,height=500)
left_frame.grid(row=0,column=0)
right_frame = Label(top_frame,width=500,height=500)
right_frame.grid(row=0,column=1)
top_frame.grid(row=0,column=0)

bottom_frame = Frame(window)
bottom_frame.grid(row=1,column=0)

threshold_label = Label(window,text=str("Adjust Threshold Level:" + str(threshold_level)),font=("Arial Bold",10))
threshold_label.grid(row=2,column=0,sticky=W,pady=10)
scrollbar = Scrollbar(window,orient="horizontal")
scrollbar.grid(row=3,column=0,sticky=E+W)


def location(x,y):
    global threshold_level,image_mode
    m = interp1d([0,1],[0,255])
    threshold_level = int(m(y))
    threshold_label.configure(text= str("Adjust Threshold Level:" + str(threshold_level)))
    scrollbar.set(y,y)
    if image_mode:
        global grimage,gthreshold,gstring,filename
        file = cv2.imread(filename)
        file=cv2.resize(file,(500,500))
        img,img_th,st= detect(file)
        updateScreen(img,img_th)
        grimage=img
        gthreshold=img_th
        gstring=st


scrollbar.config(command=location)
scrollbar.set(0.37254902,0.37254902)
def updateScreen(img,img_th):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(image= img)
    left_frame.ImageTk = img
    left_frame.configure(image = img)
    img_th = Image.fromarray(img_th)
    img_th = ImageTk.PhotoImage(image= img_th)
    right_frame.ImageTk = img_th
    right_frame.configure(image = img_th)

def detect(image):
    global threshold_level
    im = image
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    # Threshold the image
    ret, im_th = cv2.threshold(im_gray, threshold_level, 255, cv2.THRESH_BINARY_INV)
    # Find contours in the image
    ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(ctrs))
    string = ""
    if(len(ctrs) > 0):
        # Get rectangles contains each contour
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]
        (cnts,rects) = zip(*sorted(zip(ctrs,rects),key=lambda b:b[1][0],reverse=False))

        # For each rectangular region, calculate HOG features and predict
        # the digit using Linear SVM.
        for rect in rects:
            # Draw the rectangles
            cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
            # Make the rectangular region around the digit
            leng = int(rect[3] * 1.6)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
            # Resize the image
            if(roi.size == 0):
                continue
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))
            # Calculate the HOG features
            roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
            roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
            nbr = clf.predict(roi_hog_fd)
            cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
            string += str(int(nbr[0]))
    return im,im_th,string
def save_btn_clicked():
    global gstring,grimage
    if(os.path.exists("result.txt")):
        os.remove("result.txt")
    if(os.path.exists("result.jpg")):
        os.remove("result.jpg")
    cv2.imwrite("result.jpg",grimage)
    file = open("result.txt","w")
    file.write(gstring)
    file.close()
    mb.showinfo("Saved", "Result Saved!!!")
save_btn = Button(bottom_frame,text="save", command = save_btn_clicked)
save_btn.grid(row=0,column=0,padx=25,pady=25)

def clear_btn_clicked():
    global showid,image_mode
    left_frame.after_cancel(showid)
    showid = left_frame.after(10,show)
    image_mode = False

clear_btn = Button(bottom_frame,text="Clear", command = clear_btn_clicked)
clear_btn.grid(row=0,column=1,padx=25,pady=25)

def open_btn_clicked():
    global showid,gthreshold,gstring,grimage,image_mode,filename
    filename = filedialog.askopenfilename(title='open')
    file = cv2.imread(filename)
    file=cv2.resize(file,(500,500))
    left_frame.after_cancel(showid)
    img,img_th,st= detect(file)
    updateScreen(img,img_th)
    grimage=img
    gthreshold=img_th
    gstring=st
    image_mode = True
open_btn = Button(bottom_frame,text="open", command = open_btn_clicked)
open_btn.grid(row=0,column=2,padx=25,pady=25)

def show():
    global gthreshold,gstring,grimage
    ret,img = cap.read()
    img = img[80:420, 0:600]
    img,img_th,st= detect(img)
    updateScreen(img,img_th)
    grimage=img
    gthreshold=img_th
    gstring=st

    global showid
    showid = left_frame.after(10,show)
show()
window.mainloop()
