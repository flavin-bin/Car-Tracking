# Car and Pedestrian Tracking
import cv2

# Real time tracking
video=cv2.VideoCapture('drive.mp4')

# Pre-trained car classfier
class_file='car_detection.xml'

# Create car tracker
car_tracker=cv2.CascadeClassifier(class_file)

# Run forever
while True:

    # Current frame
    (read_successful, frame)=video.read()

    # If the reading of the frame is successful
    if read_successful==True:
        # Convert frame to black and white
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Tracking the cars
    cars=car_tracker.detectMultiScale(gray_frame)

    # Frame the cars found
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h), (255,0,255),2)

    # Display frame
    cv2.imshow('Car and pedestrian dectector', frame)

    # Dont autoclose
    cv2.waitKey(1)




# # Single Car image
# img_file='Car_Image.jpg'
#
# # Pre-trained car classfier
# class_file='car_detection.xml'
#
# # Create opencv image
# img=cv2.imread(img_file)
#
# # Create car tracker
# car_tracker=cv2.CascadeClassifier(class_file)
#
# # Convert to black and white (quicker but slightly less accurate)
# black_and_white=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Tracking the cars
# cars=car_tracker.detectMultiScale(black_and_white)
#
# # Frame the cars found
# for (x,y,w,h) in cars:
#     cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,255),2)
#
# # Display Image
# cv2.imshow('Car and pedestrian dectector', img)
#
# # Dont autoclose
# cv2.waitKey()

