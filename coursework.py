import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

def loadModel():  
  model_file_name =  model_name + '/frozen_inference_graph.pb'
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_file_name, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  # Загрузка меток классов объектов
  label_map = label_map_util.load_labelmap('models/research/object_detection/data/mscoco_label_map.pbtxt')
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)
  return (detection_graph, category_index)

# Загрузка изображения
def loadImage(path):
    image = Image.open(path)
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# Скачивание изображения
def downloadImage(link):
    !wget -q -O img $link
   
# Метод поиска объектов на изображении
def findOnImage(image, graph):
  with graph.as_default():
    with tf.Session() as session:
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}

      for key in ['num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Запуск поиска объектов
      output_dict = session.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

      # Преобразование выходных данных в нужный формат
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


# Основной код
(detection_graph, category_index) = loadModel()

print('Введите ссылку:')
lnk = input() 
downloadImage(lnk)
currentImage = loadImage('img')

%matplotlib inline
plt.figure(figsize=(15, 10))
plt.grid(False)
plt.imshow(currentImage)

# Запуск обнаружения объектов + Визуализация результатов поиска
output_dict = findOnImage(currentImage, detection_graph)
vis_util.visualize_boxes_and_labels_on_image_array(currentImage,
      output_dict['detection_boxes'], output_dict['detection_classes'],
      output_dict['detection_scores'], category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True, min_score_thresh = 0.6, line_thickness=5)
plt.figure(figsize=(15, 10))
plt.grid(False)
plt.axis(False)
plt.imshow(currentImage)


# https://trendymen.ru/images/article1/129272/attachments/cat.jpg