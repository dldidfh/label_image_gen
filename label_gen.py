import datetime
from config import * # global variables 
from model.video_detect import video_detection
import tensorflow as tf
import os
import cv2

class label_gen():
    def __init__(self) -> None:
        self.today_variable = str(datetime.datetime.today().year) + '_' + str(datetime.datetime.today().month) + '_' + str(datetime.datetime.today().day)
        self.path = INPUT_PATH
        self.worker_name = WORKER_NAME
        self.model_path = MODEL_PATH
        self.model = tf.saved_model.load(self.model_path)
        self.dir_num = 0 
        self.image_num = 0
        self.output_root_dir = OUTPUT_ROOT_DIR
        self.file_list = os.listdir(INPUT_PATH)
        self.analysis_location = ANALYSIS_LOCATION
        
    def data_gen(self):

        for file in self.file_list:
            video_path = INPUT_PATH + file 
            vid = cv2.VideoCapture(video_path)            
            frame_num = 0
            while True:
                return_value, frame = vid.read()
                if not return_value:
                    print(video_path, 'video end')
                    break
                if frame_num % HOW_MANY_FRAME == 0:
                    if self.image_num % HOW_MANY_IMAGES == 0 :
                        root_dir_path = self.mkdir()
                    output_path = root_dir_path + self.analysis_location + '_' + str(self.image_num)
                    boxes, _, classes, resized_frame, pad_size = video_detection('args',frame, self.model)
                    self.txt_save(boxes, output_path, pad_size, classes, resized_frame)
                    cv2.imwrite(output_path+'.jpg', frame)
                    self.image_num +=1                
                frame_num += 1

    def mkdir(self):        
        worker_len = len(self.worker_name)
        worker = self.worker_name[self.dir_num % worker_len] 
        path = OUTPUT_ROOT_DIR + worker + '_' +  self.today_variable + '_' + str(self.dir_num) + '/'
        self.dir_num +=1
        os.makedirs(path, exist_ok=True)
        return path

    def txt_save(self, boxes, output_path, pad_size, classes,resized_frame):

        x_width = pad_size[1] // 2 
        y_width = MODEL_INPUT[0] - pad_size[1] // 2
        x_height = pad_size[0] // 2 
        y_height = MODEL_INPUT[1] - pad_size[0] // 2
        resized_frame = resized_frame[0,x_width:y_width, x_height:y_height,:]        
        resized_size  = resized_frame.shape[:2]

        with open(output_path + '.txt','w') as wd:
            for i, box in enumerate(boxes):

                box[0] = box[0] - pad_size[0]//2
                box[1] = box[1] - pad_size[1]//2
                box[2] = box[2] - pad_size[0]//2
                box[3] = box[3] - pad_size[1]//2
                box_width = box[2] - box[0]
                box_height = box[3] - box[1]    
                center_x = box[2] - box_width//2
                center_y = box[3] - box_height//2

                box_width = box_width / resized_size[1]
                box_height = box_height / resized_size[0]
                center_x = center_x / resized_size[1]
                center_y = center_y / resized_size[0]

                string = "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(classes[i], center_x, center_y, box_width, box_height)
                wd.write(string)

data = label_gen()
data.data_gen()
