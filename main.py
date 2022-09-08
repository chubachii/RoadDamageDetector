import cv2
import numpy as np
import sys
import tarfile
import tensorflow as tf
import zipfile
import random

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from utils import label_map_util
from utils import visualization_utils as vis_util

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

cap = cv2.VideoCapture('input/video.AVI')
PATH_TO_CKPT =  'trainedModels/ssd_mobilenet_RoadDamageDetector.pb' 
PATH_TO_LABELS = 'trainedModels/crack_label_map.pbtxt'
NUM_CLASSES = 8

#動画サイズ取得
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#フレームレート取得
fps = cap.get(cv2.CAP_PROP_FPS)

#フォーマット指定
fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#注）グレースケールの画像を出力する場合は第5引数に0を与える
writer = cv2.VideoWriter('output/test.mp4', fmt, fps, (width, height))


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    i=1
    while True :
        print("Frame: "+ str(i))
        #フレーム情報取得
        ret, img = cap.read()
        
        #動画が終われば処理終了
        if ret == False:
            break
        
        #image_np = load_image_into_numpy_array(img)
        image_np = img

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            min_score_thresh=0.3,
            use_normalized_coordinates=True,
            line_thickness=8)

        
        #動画書き込み
        writer.write(image_np)
        
        i +=1
        

cap.release()
#これを忘れるとプログラムが出力ファイルを開きっぱなしになる
writer.release()