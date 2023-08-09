import cv2


# Create our body classifier
face_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret,img = cap.read()

    #Convert Each Frame into Grayscale
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    faces = face_cascade.detectMultiScale(gray)
    print(faces)
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in faces:
       cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()





