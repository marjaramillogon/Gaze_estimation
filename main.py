import mediapipe as mp
import cv2
import gaze
import pickle

mp_face_mesh = mp.solutions.face_mesh  # initialize the face mesh model

# camera stream:
cap = cv2.VideoCapture(0)  # chose camera index (try 1, 2, 3)

video_test = cv2.VideoCapture('sacadich.mp4') 
   
left_pupil_vec_x=[]
left_pupil_vec_y=[]
right_pupil_vec_x=[]
right_pupil_vec_y=[]

cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

with mp_face_mesh.FaceMesh(
        max_num_faces=1,  # number of faces to track in each frame
        refine_landmarks=True,  # includes iris landmarks in the face mesh model
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while video_test.isOpened():
        ret, frame_vid = video_test.read() 
        if ret == True: 
            success, image = cap.read()
            cv2.imshow("window", frame_vid) 

            if not success:  # no frame input
                print("Ignoring empty camera frame.")
                break
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # frame to RGB for the face-mesh model
            results = face_mesh.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # frame back to BGR for OpenCV

            if results.multi_face_landmarks:
                left_pupil, right_pupil= gaze.gaze(image, results.multi_face_landmarks[0])  # gaze estimation

                left_pupil_vec_x.append(left_pupil[0])
                left_pupil_vec_y.append(left_pupil[1])
                right_pupil_vec_x.append(right_pupil[0])
                right_pupil_vec_y.append(right_pupil[1])

            #cv2.imshow('output window', image)
            if cv2.waitKey(2) & 0xFF == 27:
                break
        else:
            name_coords='coords.pkl'                                  
            print('finish')
            with open(name_coords, 'wb') as folder:
                pickle.dump([left_pupil_vec_x,left_pupil_vec_y,right_pupil_vec_x,right_pupil_vec_y],folder)
            break

    cap.release()
    video_test.release()

    cv2.destroyAllWindows()

