#General Libraries..
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import pprint
import csv

#pose_utils.py
from pose_utils import parse_output, draw_kps, join_point

model_path = 'models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = input_details[0]['dtype'] == np.float32

video_path = 'videos/Guinness World Records - Most Jump Rope Skips in 30 Seconds on One Foot.mp4'
video = cv2.VideoCapture(video_path)

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
y_coords = []

while(video.isOpened()):
    # Start timer (for calculating frame rate)
    current_count=0
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    ret, frame1 = video.read()

    if ret == True:
        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame, (width, height))
        input_frame = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_frame = (np.float32(input_frame) - 127.5) / 127.5

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_frame)
        interpreter.invoke()   

        #get data from interpreter
        frame_output_data = interpreter.get_tensor(output_details[0]['index'])
        frame_offset_data = interpreter.get_tensor(output_details[1]['index'])

        #remove the extra dimension we added initially
        frame_heatmaps = np.squeeze(frame_output_data)
        frame_offset = np.squeeze(frame_offset_data)

        frame_show = np.squeeze((input_frame.copy()*127.5+127.5)/255.0)
        frame_show = np.array(frame_show*255, np.uint8)
        frame_kps = parse_output(frame_heatmaps, frame_offset, 0.3)
        final_frame = draw_kps(frame_show.copy(), frame_kps)
        #print(image_kps) Prints all key-points.. (Successfully shows keypoints.)
        
        kps_list = frame_kps.tolist()
        y_coords.append([y_coords[0] for y_coords in kps_list])

        '''black_pose = np.zeros_like(frame_show)
        join_point(black_pose, frame_kps[:, :2])
        cv2.imshow('stickman only', black_pose)'''

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        # Draw framerate in corner of frame
        cv2.putText(final_frame,'FPS: {0:.2f}'.format(frame_rate_calc),(15,25),cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,0),1,cv2.LINE_AA)
        resize = cv2.resize(final_frame, (640, 480), interpolation = cv2.INTER_LINEAR)
        cv2.imshow('Keypoints',resize)
        # Press 'q' to quit
        if cv2.waitKey(1)==ord('q'):
            break
    else:
        break
#pprint.pprint(y_coords)
'''with open('output/posture.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(y_coords)'''
cv2.destroyAllWindows()