import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math

from pose_utils import parse_output, draw_kps, join_point

model_path = 'models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

#print(input_details, output_details) #Testing if model is working

image_src = cv2.imread('photos/woman2.jpg')
image = cv2.resize(image_src, (width, height))

#Test if the original image and resized image is displayed (Is displaying)
#cv2.imshow('image original', image)

image_input = np.expand_dims(image.copy(), axis=0)
floating_model = input_details[0]['dtype'] == np.float32

if floating_model:
    image_input = (np.float32(image_input)-127.5)/127.5

interpreter.set_tensor(input_details[0]['index'], image_input)
interpreter.invoke()

#get data from interpreter
image_output_data = interpreter.get_tensor(output_details[0]['index'])
image_offset_data = interpreter.get_tensor(output_details[1]['index'])

#remove the extra dimension we added initially
image_heatmaps = np.squeeze(image_output_data)
image_offset = np.squeeze(image_offset_data)

#test shapes of above output..
#print("image_heatmaps_shape:", image_heatmaps.shape)#(9,9,17)
#print("image_offsets_shape", image_offset.shape)#(9,9,34)

image_show = np.squeeze((image_input.copy()*127.5+127.5)/255.0)
image_show = np.array(image_show*255, np.uint8)
image_kps = parse_output(image_heatmaps, image_offset, 0.3)
#print(image_kps) #Prints all key-points.. (Successfully shows keypoints.)
final_image = draw_kps(image_show.copy(), image_kps)
resize = cv2.resize(final_image, (640, 480), interpolation = cv2.INTER_LINEAR)

cv2.imshow('Keypoints',resize)
#print(image_kps)
'''kps_list = image_kps.tolist()
print([y_coords[0] for y_coords in kps_list])'''
'''y_coords = image_kps[:, 0]
x_coords = image_kps[:, 1]'''
#print(x_coords, y_coords)
'''black_pose = np.zeros_like(image_show)
resultant = join_point(black_pose, image_kps[:, :2])
cv2.imshow('stickman only', resultant)'''

cv2.waitKey(0)
cv2.destroyAllWindows()