from Detector import *
import os

from main import Dectector
def main():
    # videoPath="C:/Python/OpenCV/Projects/OD/City_Street.mp4"    # Video file
    videoPath=0     # Camera
    configPath=os.path.join("model_data","C:/Python/OpenCV/Projects/OD/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")   # Change file path
    modelPath=os.path.join("model_data","C:/Python/OpenCV/Projects/OD/frozen_inference_graph (1).pb")   # Change file path
    classesPath=os.path.join("model_data","C:/Python/OpenCV/Projects/OD/coco.names.txt")    # Change file path
    detector=Dectector(videoPath,configPath,modelPath,classesPath)
    detector.onVideo()
if __name__=='__main__':
    main()
