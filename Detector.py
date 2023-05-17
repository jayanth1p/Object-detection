import cv2
import numpy as np
import time
class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = 0
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = 'E:\\project\\ML\\vide-det\\model_data\\coco.names'
        
        
        self.net = cv2.dnn_DetectionModel('E:\\project\\ML\\vide-det\\model_data\\frozen_inference_graph.pb', 'E:\\project\\ML\\vide-det\\model_data\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
        self.net.setInputSize(320, 320)
        self.net.setInputScale (1.0/127.5)
        self.net.setInputMean ((127.5, 127.5, 127.5))
        self.net.setInputSwapRB (True) 
        self.readClasses()
    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classeslist= f.read().splitlines()
            
        self.classeslist.insert(0,' __Background__')
        print(self.classeslist)
        
    def onVideo(self):
        print("start")
        cap = cv2.VideoCapture ('C:\\Users\\pedad\\Downloads\\dronvid.mp4')
        if (cap.isOpened()==False):
            print("Error opening file...")
        # return
        (success, image) = cap.read()
        
        
        out=cv2.VideoWriter('dron-det.mp4',cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),int(cap.get(cv2.CAP_PROP_FPS)),(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),False)
        print(out)
        while success:
            print("jayanth")
            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold = 0.4)
            bboxs =list (bboxs)
            confidences = list (np.array(confidences).reshape (1,-1)[0])
            confidences = list (map(float, confidences))
            bboxIdx = cv2.dnn.NMSBoxes (bboxs, confidences, score_threshold = 0.5, nms_threshold = 0.2)
            if len (bboxIdx) != 0:
                for i in range(0, len(bboxIdx)):
                    bbox = bboxs [np. squeeze (bboxIdx[i])]
                    classConfidence = confidences [np. squeeze (bboxIdx[i])]
                    classLabelID = np. squeeze (classLabelIDs [np. squeeze (bboxIdx[i])])
                    classLabel = self.classeslist[classLabelID]
                    
                    displayText = "{}: {: .2f}" .format(classLabel, classConfidence)
                    
                    x,y,w,h = bbox
                    cv2.rectangle(image, (x,y), (x+w, y+h), color=(0,255,0), thickness=2)
                    cv2.putText(image, displayText, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)
                    
            cv2.imshow("Result", image)
            
            out.write(image)
            # print(image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            (success, image) = cap.read()
        
        print("completed")
        out.release()
        cap.release()
        
        cv2.destroyAllWindows ()

            
            
        