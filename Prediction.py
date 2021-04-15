import numpy as np
import tensorflow as tf
import os
import sys
import cv2
from research.object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from Preprocessing import EncodeDecode



class ImagePredict:
    def __init__(self,imagename, modelpath):
        self.IMAGE = imagename
        self.MODEL_NAME = modelpath

        sys.path.append("..")
        CWD_path = os.getcwd()

        self.PATH_TO_CKPT = os.path.join(CWD_path, self.MODEL_NAME,'frozen_inference_graph.pb')
        self.PATH_TO_LABELS = os.path.join(CWD_path, 'research/data','labelmap.pbtxt')
        self.PATH_TO_IMAGE = os.path.join(CWD_path,'research',self.IMAGE)

        self.NUM_OF_CLASSES = 1

        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map,max_num_classes=self.NUM_OF_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        self.class_names_mapping = {1: "Helmets"}
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            # Input tensor is the image
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

            # Output tensors are the detection boxes, scores, and classes
            # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represents level of confidence for each of the objects.
            # The score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

            # Number of objects detected
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')




    def FinalPrediction(self):


        sess = tf.Session(graph=self.detection_graph)
        image = cv2.imread(self.PATH_TO_IMAGE)
        image_expanded = np.expand_dims(image, axis=0)
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})

        print("i am boxes values",boxes)
        print("i am score value",scores)
        print("i am classes value", classes)
        print("i am total no of detections",num)

        flattened_scores = scores.flatten()
        scores_index = []
        for index in range(0,len(flattened_scores)):
            if flattened_scores[index] > 0.60:
                scores_index.append(index)

        print("i am index for score after threshold", scores_index)

        flattened_classes = classes.flatten()
        top_scores = [flattened_classes[id] for id in scores_index]

        print("i am value of top_scores", top_scores)

        final_class_names = [self.class_names_mapping[i] for i in top_scores]

        top_confidence_scores = []
        for x in flattened_scores:
            if x > 0.50:
                top_confidence_scores.append(x)

        print("i am top confidence scores",top_confidence_scores)

        new_boxes = boxes.reshape(300, 4)
        max_boxes_to_draw = new_boxes.shape[0]
        min_score_thresh = .30


        finaloutputlist = []

        for (names,score,index) in zip(final_class_names, top_confidence_scores, range(min(max_boxes_to_draw,new_boxes.shape[0]))):
            output = {}
            output["className"] = names
            output["confidence"] = str(score)

            if flattened_scores is None or flattened_scores[index] > min_score_thresh:
                val = list(new_boxes[index])
                output["yMin"] = str(val[0])
                output["xMin"] = str(val[1])
                output["yMax"] = str(val[2])
                output["xMax"] = str(val[3])
                finaloutputlist.append(output)

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.40)
        output_filename = 'output4.jpg'
        cv2.imwrite(output_filename, image)
        opencodedbase64 = EncodeDecode.encodeIntoBase64("output4.jpg")

        finaloutputlist.append({"image": opencodedbase64.decode('utf-8')})
        return finaloutputlist











