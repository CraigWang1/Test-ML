#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
from tensorflow.python.platform import gfile

def resize_image(image, image_size):
  """Resize image so that longest side length = image_size, preserves aspect ratio"""
  image_height, image_width = image.shape[:2]
  if image_height > image_width:
    scale = image_size / image_height
    resized_height = image_size
    resized_width = int(image_width * scale)
  else:
    scale = image_size / image_width
    resized_height = int(image_height * scale)
    resized_width = image_size

  resized_image = cv2.resize(image, (resized_width, resized_height))
  return resized_image, scale

def preprocess_image(image, image_size):
  """Helper function to process image to input to model. Expects image with longest side = 512 pixels"""
  image_height, image_width = image.shape[:2]

  new_image = np.ones((image_size, image_size, 3), dtype=np.float32) * 128.
  new_image[0:image_height, 0:image_width] = image.astype(np.float32)
  new_image /= 255.

  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]

  for i in range(3):
    new_image[..., i] -= mean[i]
    new_image[..., i] /= std[i]

  return new_image

def inference_vid(model_path, input_vid, output_vid='/content/result.mp4', phi=0, weighted_bifpn=False, classes=None, score_threshold=0.5, draw=True, fps=30):
  """Function to inference on videos using pb frozen graph model"""
  image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
  image_size = image_sizes[phi]
  num_classes = len(classes)
  colors = [(0,255,255)]  #yellow

  tf.reset_default_graph()
  sess = tf.Session()
  with tf.gfile.GFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

  tensor_input = sess.graph.get_tensor_by_name('input_1:0')

  output_boxes = sess.graph.get_tensor_by_name('filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0')
  output_scores = sess.graph.get_tensor_by_name('filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0')
  output_labels = sess.graph.get_tensor_by_name('filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0')

  cap = cv2.VideoCapture(input_vid)
  out = cv2.VideoWriter(output_vid,  cv2.VideoWriter_fourcc(*"XVID"), fps,
                        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
  while cap.isOpened():
    start = time.time()
    flag, image = cap.read()
    if not flag:
      break

    image, scale = resize_image(image, image_size)
    input_image = image[:, :, ::-1]   #convert from bgr to rgb (opencv reads bgr, model needs rgb)
    h, w = input_image.shape[:2]

    input_image = preprocess_image(input_image, image_size)
    # run network
    boxes, scores, labels = sess.run([output_boxes, output_scores, output_labels], {tensor_input:[input_image]})
    # Don't rescale boxes, so that we output the downscaled image like the cpp file
    # boxes /= scale
    boxes[0, :, 0] = np.clip(boxes[0, :, 0], 0, w - 1)
    boxes[0, :, 1] = np.clip(boxes[0, :, 1], 0, h - 1)
    boxes[0, :, 2] = np.clip(boxes[0, :, 2], 0, w - 1)
    boxes[0, :, 3] = np.clip(boxes[0, :, 3], 0, h - 1)

    # select indices which have a score above the threshold
    indices = np.where(scores[0, :] > score_threshold)[0]

    # select those detections
    boxes = boxes[0, indices]
    scores = scores[0, indices]
    labels = labels[0, indices]

    # draw the box
    for box, score, label in zip(boxes, scores, labels):
      xmin = int(round(box[0]))
      ymin = int(round(box[1]))
      xmax = int(round(box[2]))
      ymax = int(round(box[3]))
      score = '{:.4f}'.format(score)
      class_id = int(label)
      color = colors[class_id]
      class_name = classes[class_id]
      label = '-'.join([class_name, score])
      ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
      cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 3)
      cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
      cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    out.write(image)
    #print fps
    print("FPS:", 1/(time.time() - start))
  cap.release()
  out.release()

def inference_pic(model_path, phi=0, weighted_bifpn=False, classes=None, score_threshold=0.5, save=True):
  """Function to inference on pictures"""
  image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
  image_size = image_sizes[phi]
  num_classes = len(classes)
  colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]

  tf.reset_default_graph()
  sess = tf.Session()
  with tf.gfile.GFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

  tensor_input = sess.graph.get_tensor_by_name('input_1:0')

  output_boxes = sess.graph.get_tensor_by_name('filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0')
  output_scores = sess.graph.get_tensor_by_name('filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0')
  output_labels = sess.graph.get_tensor_by_name('filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0')

  # loop to keep inferencing on user's images
  while True:
    image_path = input("Path to image: ")
    image = cv2.imread(image_path)
    src_image = image.copy()
    image = image[:, :, ::-1]   #goes from bgr to rgb
    h, w = image.shape[:2]

    image, scale = preprocess_image(image, image_size=image_size)
    # run network
    start = time.time()
    boxes, scores, labels = sess.run([output_boxes, output_scores, output_labels], {tensor_input:[image]})
    print(time.time() - start)
    boxes /= scale
    boxes[0, :, 0] = np.clip(boxes[0, :, 0], 0, w - 1)
    boxes[0, :, 1] = np.clip(boxes[0, :, 1], 0, h - 1)
    boxes[0, :, 2] = np.clip(boxes[0, :, 2], 0, w - 1)
    boxes[0, :, 3] = np.clip(boxes[0, :, 3], 0, h - 1)

    # select indices which have a score above the threshold
    indices = np.where(scores[0, :] > score_threshold)[0]

    # select those detections
    boxes = boxes[0, indices]
    scores = scores[0, indices]
    labels = labels[0, indices]

    # visualize the outputs
    for box, score, label in zip(boxes, scores, labels):
      xmin = int(round(box[0]))
      ymin = int(round(box[1]))
      xmax = int(round(box[2]))
      ymax = int(round(box[3]))
      score = '{:.4f}'.format(score)
      class_id = int(label)
      color = colors[class_id]
      class_name = classes[class_id]
      label = '-'.join([class_name, score])
      ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
      cv2.rectangle(src_image, (xmin, ymin), (xmax, ymax), color, 3)
      cv2.rectangle(src_image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
      cv2.putText(src_image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # if specified to write output to image file
    if save:  
      cv2.imwrite('/content/result.jpg', src_image)
    # display image, but have to convert to rgb for matplotlib
    plt.imshow(cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB))  
    plt.show()