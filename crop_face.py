import cv2
import os

def facecrop(img1):
    facedata = r"G:\study\Research projecy\CNN\haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)
    #image=r'G:\study\Research projecy\CNN\amey\amey.2.1.jpg'
   # img = cv2.imread(image)
    img = cv2.imread(r'C:\Users\Amey\Desktop\amey.2.1.jpg')
    img = cv2.resize(img, (200, 200), interpolation = cv2.INTER_AREA)
    facedata = r"G:\study\Research projecy\CNN\haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)
    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)
    faces = cascade.detectMultiScale(miniframe)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))
        sub_face = img[y:y+h, x:x+w]
   img=sub_face     
   img = cv2.resize(img, (150, 150), interpolation = cv2.INTER_AREA)
   cv2.imshow("fasf",img)
   cv2.waitKey(1)
   cv2.destroyAllWindows()
     #fname, ext = os.path.splitext(r'C:\Users\Amey\Desktop\ActiOn_3.jpg')
        #cv2.imwrite(fname+"_cropped_"+ext, sub_face)
    return


img = cv2.imread(r'C:\Users\Amey\Desktop\ActiOn_3.jpg')
facecrop(img)
size=150
def crop(img):
    facedata = r"G:\study\Research projecy\CNN\haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)
    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)
    faces = cascade.detectMultiScale(miniframe)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))
        sub_face = img[y:y+h, x:x+w] 
    return sub_face
img = cv2.imread(r'C:\Users\Amey\Desktop\ActiOn_5.jpg')   
img= crop(img)
img = cv2.resize(img, (size, size), interpolation = cv2.INTER_AREA)
cv2.imshow("fasf",img)
cv2.waitKey(1)
cv2.destroyAllWindows()
