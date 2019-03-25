#############################################################
#part 1
import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier(r'G:\study\Research projecy\facial_recognition\haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while(True):

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite(r"G:\study\Research projecy\facial_recognition\dataset\User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.putText(gray, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('image', img)
        print("Face found {}".format(count))
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 100: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


###########################################################
#part 2
import os

import numpy as np

import cv2
cv2.__version__
from PIL import Image # For face recognition we will the the LBPH Face Recognizer 


recognizer = cv2.face.LBPHFaceRecognizer_create()
path=r"G:\study\Research projecy\facial_recognition\dataset"

def getImagesWithID(path):

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]   

 # print image_path   

 #getImagesWithID(path)

    faces = []

    IDs = []

    for imagePath in imagePaths:      

  # Read the image and convert to grayscale

        facesImg = Image.open(imagePath).convert('L')

        faceNP = np.array(facesImg, 'uint8')

        # Get the label of the image

        ID= int(os.path.split(imagePath)[-1].split(".")[1])

         # Detect the face in the image

        faces.append(faceNP)

        IDs.append(ID)

        cv2.imshow("Adding faces for traning",faceNP)

        cv2.waitKey(10)

    return np.array(IDs), faces

Ids,faces  = getImagesWithID(path)

recognizer.train(faces,Ids)

recognizer.save(r'G:\study\Research projecy\facial_recognition\trainer.yml')

cv2.destroyAllWindows()

##################################################
#Part 3
import cv2
import numpy as np
import os 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(r'G:\study\Research projecy\facial_recognition\trainer.yml')
cascadePath = r"G:\study\Research projecy\facial_recognition\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['','Amey','himanshu'] 

# Initialize and start realtime video capture
#'http://192.168.0.106/web/admin.html'
#'http://admin:camera1@192.168.0.106/web/admin.html'
#'http://admin:camera1@192.168.0.106/tmpfs/auto.jpg'
cam =cv2.VideoCapture(0)
cam.read()
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
while True:
    ret, img =cam.read()
    #img = cv2.flip(img,-1) # Flip vertically

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            print(id)
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,0), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()







