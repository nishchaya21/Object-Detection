from Detector import *
import os

from main import Dectector
def main():
    # videoPath="C:/Users/nishc/Downloads/City_Street.mp4"    # Video file
    videoPath=0     # Camera
    configPath=os.path.join("model_data","C:/Python/OpenCV/Projects/OD/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath=os.path.join("model_data","C:/Python/OpenCV/Projects/OD/frozen_inference_graph (1).pb")
    classesPath=os.path.join("model_data","C:/Python/OpenCV/Projects/OD/coco.names.txt")
    detector=Dectector(videoPath,configPath,modelPath,classesPath)
    detector.onVideo()
if __name__=='__main__':
    main()