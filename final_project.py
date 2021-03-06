import argparse
import contextlib
import threading
import time
from PIL import Image
from PIL import ImageDraw, ImageFont
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import list_edge_tpus
from pycoral.utils.edgetpu import make_interpreter
import numpy as np
import cv2

from utils import _keypoints_and_edges_for_display
label = []
f = open('pose_labels.txt','r')
for line in f.readlines():
  line=line.strip('\n')
  line=line.strip(' ')
  #print(line)
  label.append(line)
f.close()
def draw_objects(draw, objs):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='red')
    #draw.text((bbox.xmin + 10, bbox.ymin + 10),
    #          '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
    #          fill='red')
def non_max_suppression(objects, threshold):
  """Returns a list of indexes of objects passing the NMS.

  Args:
    objects: result candidates.
    threshold: the threshold of overlapping IoU to merge the boxes.
..
  Returns:
    A list of indexes containings the objects that pass the NMS.
  """
  if len(objects) == 1:
    return [0]

  boxes = np.array([o.bbox for o in objects])
  xmins = boxes[:, 0]
  ymins = boxes[:, 1]
  xmaxs = boxes[:, 2]
  ymaxs = boxes[:, 3]

  areas = (xmaxs - xmins) * (ymaxs - ymins)
  scores = [o.score for o in objects]
  idxs = np.argsort(scores)

  selected_idxs = []
  while idxs.size != 0:

    selected_idx = idxs[-1]
    selected_idxs.append(selected_idx)

    overlapped_xmins = np.maximum(xmins[selected_idx], xmins[idxs[:-1]])
    overlapped_ymins = np.maximum(ymins[selected_idx], ymins[idxs[:-1]])
    overlapped_xmaxs = np.minimum(xmaxs[selected_idx], xmaxs[idxs[:-1]])
    overlapped_ymaxs = np.minimum(ymaxs[selected_idx], ymaxs[idxs[:-1]])

    w = np.maximum(0, overlapped_xmaxs - overlapped_xmins)
    h = np.maximum(0, overlapped_ymaxs - overlapped_ymins)

    intersections = w * h
    unions = areas[idxs[:-1]] + areas[selected_idx] - intersections
    ious = intersections / unions

    idxs = np.delete(
        idxs, np.concatenate(([len(idxs) - 1], np.where(ious > threshold)[0])))

  return selected_idxs
_NUM_KEYPOINTS = 17
def run_two_models_two_tpus(detection_model,pose_estimation_model,video_name,
                            num_inferences):
  results=[]
  num_people=[]
  index = 0
  def detection_job(detection_model, video_name, num_inferences):
    """Runs detection job."""
    interpreter = make_interpreter(detection_model, device=':1')
    
    interpreter.allocate_tensors()
    
    cap = cv2.VideoCapture(video_name)
    #image = Image.open(image_name)
    while True:
      ret,image = cap.read()
      if not ret:
        break
      image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
      _, scale = common.set_resized_input(
          interpreter, image.size,
          lambda size: image.resize(size, Image.NEAREST))
    
      interpreter.invoke()
      objs=detect.get_objects(interpreter, score_threshold=0.5, image_scale=scale)
      if len(objs)==0:
        continue
      idx = non_max_suppression(objs,0.1)
      count=0
      for i in idx:
        count+=1
        results.append(objs[i])
      num_people.append(count)
    cap.release()
  def pose_estimation_job(pose_estimation_model, video_name, num_inferences,results):
    """Runs pose_estimation job."""
    interpreter = make_interpreter(pose_estimation_model, device=':0')
    interpreter.allocate_tensors()
    interpreter_classify = make_interpreter('pose_classifier_quant_edgetpu.tflite',device=':0')
    interpreter_classify.allocate_tensors()
    #img = Image.open(image_name)
    cap = cv2.VideoCapture(video_name)
    ret,test=cap.read()
    test = Image.fromarray(cv2.cvtColor(test,cv2.COLOR_BGR2RGB))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('result.mp4', fourcc, 30, test.size)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
    global index
    index=0
    frame_number=0
    while True:
      ret,image = cap.read()
      if not ret:
        break
      while len(results)==0 :
        time.sleep(0.0000000000000001)
      size = len(results)
      
      print(size)
      image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    #imgs = []
    #for res in results:
        #bbox = res.bbox
        #imgs.append(img.crop((bbox.xmin,bbox.ymin,bbox.xmax,bbox.ymax)))
      start=index
      for k in range(start,start+num_people[frame_number]):
        index=index+1
        bbox=results[k].bbox
        cropped_img = image.crop((bbox.xmin,bbox.ymin,bbox.xmax,bbox.ymax))
        
        resized_img = cropped_img.resize(common.input_size(interpreter), Image.ANTIALIAS)
        common.set_input(interpreter, resized_img)
        interpreter.invoke()


        
        keypoints_with_scores = common.output_tensor(interpreter, 0).copy()
        pose = keypoints_with_scores.reshape(_NUM_KEYPOINTS, 3)
        pose_classify=[]
        params = common.input_details(interpreter_classify, 'quantization_parameters')
        
        for i in range(17):
          for j in range(3):
            if j==2:
              pose_classify.append(pose[i][j])
            else:
              pose_classify.append(pose[i][j]/params['scales'])
        if len(pose_classify)==0:
          continue
        pose_classify=np.array(pose_classify)
        pose_classify = pose_classify.reshape((1,17,3))
        pose_classify = pose_classify.astype(np.uint8)
        common.set_input(interpreter_classify,pose_classify)
        params = common.input_details(interpreter_classify, 'quantization_parameters')
        scale = params['scales']
        zero_point = params['zero_points']
        #print(scale)
        #print(zero_point)
        interpreter_classify.invoke()
        classes = classify.get_classes(interpreter_classify, 1, 0.0)
        for c in classes:
          print(c)
        draw = ImageDraw.Draw(cropped_img)
        width, height = cropped_img.size
        draw.rectangle([(0.01,0.01), (width-0.01, height-0.01)],outline='green', width=2)
        (keypoint_locs, keypoint_edges, edge_colors)  = _keypoints_and_edges_for_display(keypoints_with_scores, height, width)
        
        for i, edge in enumerate(keypoint_edges):
          if i < _NUM_KEYPOINTS:
            draw.ellipse(
                xy=[
                    pose[i][1] * width - 2, pose[i][0] * height - 2,
                    pose[i][1] * width + 2, pose[i][0] * height + 2
                ],
              fill=(255, 0, 0))
          draw.line(edge)
        draw.rectangle((2,2, 90, 30), fill='green')
        #print(classes[0].id)
        draw.text((5,5), label[classes[0].id], align ="left", font=font) 
        image.paste(cropped_img,(bbox.xmin,bbox.ymin))
        
  
      image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
      out.write(image)
      frame_number+=1
    cap.release()

  start_time = time.perf_counter()

  detection_thread = threading.Thread(
      target=detection_job, args=(detection_model, video_name, num_inferences))
  pose_estimation_thread = threading.Thread(
      target=pose_estimation_job,
      args=(pose_estimation_model, video_name, num_inferences,results))
  detection_thread.start()
  pose_estimation_thread.start()
  detection_thread.join()
  pose_estimation_thread.join()
  return time.perf_counter() - start_time
  
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--pose_estimation_model',
      help='Path of classification model.',
      required=True)
    parser.add_argument(
      '--video',
      help='input demo video',
      required=True)
    parser.add_argument(
      '--detection_model', help='Path of detection model.', required=True)
    parser.add_argument('--image', help='Path of the image.')
    parser.add_argument(
      '--num_inferences',
      help='Number of inferences to run.',
      type=int,
      default=5)
    parser.add_argument(
      '--batch_size',
      help='Runs one model batch_size times before switching to the other.',
      type=int,
      default=1)
    args = parser.parse_args()
    time = run_two_models_two_tpus(args.detection_model,args.pose_estimation_model,args.video,args.num_inferences)
    print(time)
