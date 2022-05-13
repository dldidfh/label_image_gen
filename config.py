import os

ROOT_PATH = 'D:/test_videos/bangkok/'
RTSP_URL = 'rtsp://'
ANALYSIS_LOCATION = 'YoungIn_VDS'  # 이미지가 저장되는 폴더 이름 
# ANALYSIS_LOCATION = 'Gungiyung_night_data_18_to_06' 
ROI_SET = False # ROI 설정이 필요할 경우 True 설정 
DETECTING = False # 모델을 통해 라벨링 하기 전에 검출을 진행하여 txt 를 저장함 
INPUT_PATH = [ROOT_PATH]
# for folder in os.listdir(ROOT_PATH):
#     dir_path2 = os.path.join(ROOT_PATH, folder)
#     INPUT_PATH.append(dir_path2 )
            


OUTPUT_ROOT_DIR = './output/'
WORKER_NAME = ['YoungIn_VDS'] # 이미지 파일이 저장되는 이름 

HOW_MANY_IMAGES = 300 # 하나의 폴더에 몇장의 이미지를 넣을지 
HOW_MANY_FRAME = 900 # 몇프레임마다 이미지를 저장할 지 15FPS = 1초 

VUE_THRESHOLD = 10 # 아무 차량이 없을 때를 캡쳐하는 것을 방지하기 위해 배경추출을 통해 움직인 객체가 있을 때만 이미지 저장 

with open('./last_file_number.txt', 'r') as rd:
    lines = rd.readlines()
    CURRENT_IMAGE_NUM = int(lines[0].strip().split()[-1]) + 1 # last_file_number.txt 파일에 지금까지 저장된 이미지의 개수와 디렉토리 개수를 저장하여 다음 실행 때 파일명, 폴더명이 중복되지 않게 함 
    CURRENT_DIR_NUM = int(lines[1].strip().split()[-1] ) 






MODEL_PATH = './model/best_model_p5_252_0.682/1'
MODEL_INPUT = (608,608)

HAS_TIME_CONDITION = False
REGEX_FLAG = [180000, 60000]