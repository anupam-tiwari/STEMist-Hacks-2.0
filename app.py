from streamlit_webrtc import webrtc_streamer
import av
import cv2
import mediapipe as mp 
from pygame import mixer
import time

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

mixer.init()
drum_clap = mixer.Sound('batterrm.wav')
drum_snare = mixer.Sound('button-2.ogg')

rect_x = 200  # x-coordinate of the top-left corner of the rectangle
rect_y = 100  # y-coordinate of the top-left corner of the rectangle
rect_width = 150  # Width of the rectangle
rect_height = 150  # Height of the rectangle




def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 255, 0), 2)
    
    results = hands.process(img)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)
            
            
            
            # Get the coordinates of the forefinger (index finger)
            if len(hand_landmarks.landmark) > 8:  # Check if the landmark list has the index finger landmark (ID 8)
                index_finger_landmark = hand_landmarks.landmark[8]
                image_height, image_width, _ = img.shape  # Get the image's height and width for scaling
                x_pixel, y_pixel = int(index_finger_landmark.x * image_width), int(index_finger_landmark.y * image_height)
                cv2.circle(img, (x_pixel, y_pixel), 5, (0, 0, 255), -1)
            
            if rect_x <= x_pixel <= rect_x + rect_width and rect_y <= y_pixel <= rect_y + rect_height:
                drum_clap.play()
                time.sleep(1)
            else:
                print("x_pixel and y_pixel are outside the rectangle!")
    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=video_frame_callback)