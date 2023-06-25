
import tensorflow as tf
import cv2
from layers import L1Dist
import os
import numpy as np
import face_recognition

# load model in with custom L1 distance layer
my_model= tf.keras.models.load_model('siamesemodel', custom_objects={'L1dist':L1Dist}) 
capture = cv2.VideoCapture(1)  # note: 0 is default

#preprocess the known faces
known_faces = []
known_faces_names = []
for image in os.listdir(os.path.join('ppl')):

    # read images in
    img = cv2.imread(os.path.join('ppl',image))
    # use face_recigniton to find the coordinates of the top, bottom, left and right sides of the face
    face_loc = face_recognition.face_locations(img)
    # slice the image to be the face
    known_face = img[int(face_loc[0][0]):int(face_loc[0][2]), int(face_loc[0][3]):int(face_loc[0][1]), :]
    # convert the slice to a tensor to be used in the model
    known_face = tf.convert_to_tensor(known_face)
    # resize to 100x100 and scale to be between 1 and 0
    known_face = tf.image.resize(known_face, (100,100))
    known_face = known_face/255.0
    # append to known faces list to compare with faces in frame
    known_faces.append(known_face)

    # get the filename and use it as the face's name
    name = os.path.basename(image)
    known_faces_names.append(name)
    
# define necessary lists 
#face_names = []
faces = []
face_locations = []
face_names = []

process_this_frame = True

while True:
    #grab a single frame of video
    ret, frame = capture.read()

    if process_this_frame:
        # resize frame to 1/4 size for faster recognition:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        face_locations = []

        #find all the faces in the cam feed:
        face_locations = face_recognition.face_locations(small_frame)  #coordinates of faces

        faces = []

        # loop through the face locations in frame
        for index, location in enumerate(face_locations):
            # store the face values, scaled by 4 from small_frame
            top = 4 * int(face_locations[index][0])
            right = 4 * int(face_locations[index][1])
            bottom = 4 * int(face_locations[index][2])
            left = 4 * int(face_locations[index][3])

            # slice the frame
            face_frame = frame[top:bottom+(250-bottom+top), left:right+(250-right+left), :]
            # convert to tensor to work with model
            face_frame = tf.convert_to_tensor(face_frame)
            # resize and scale
            face_frame = tf.image.resize(face_frame, (100,100))
            face_frame = face_frame/255.0
            # store in faces list
            faces.append(face_frame)
                  
        # reset lists in case frame environment changes
        matches = []
        face_names = []

        # loop through the faces in frame and compare them to each face in known_faces
        for face in faces:
            matches = []
            for image in known_faces:
                # use model to calculate similarity
                match = my_model.predict(list(np.expand_dims([face, image], axis = 1 )), verbose =0)
                # store each prediction
                matches.append(match)
                name="unknown"

            # the max value in matches will represent the best match for the face in frame
            max_value = max(matches)
            if max_value>=0.6:
                name = known_faces_names[matches.index(max_value)]
            face_names.append(name)

    process_this_frame = not process_this_frame
    
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
capture.release()
cv2.destroyAllWindows()