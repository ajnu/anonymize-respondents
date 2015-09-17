import cv2
import sys

# Get user supplied values
imagePath = "DSC_0088.JPG"
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

result_image = image.copy()

faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
        )

print "Found {0} faces!".format(len(faces))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#can omit this
if len(faces) != 0:
    for f in faces:
        x, y, w, h = [v for v in f]


if len(faces) != 0:
    for f in faces:
        x, y, w, h = [v for v in f]
        cv2.rectangle(image, (x,y), (x+w,y+h), (255,255,0), 5)
        sub_face = image[y:y+h, x:x+w]
        sub_face = cv2.GaussianBlur(sub_face,(23, 23), 30)
        result_image[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face
        face_file_name = "./face_" + str(y) + ".jpg"
        cv2.imwrite(face_file_name, sub_face)


cv2.imwrite("./result.png", result_image)
