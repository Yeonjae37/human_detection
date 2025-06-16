import os
import time
import cv2
import torch
from flask import Flask
from human_detection.models.detector import HumanDetector
from human_detection.models.tracker import HumanTracker
from human_detection.utils.visualization import draw_timestamp, process_masks, draw_detection_boxes
from human_detection.web.routes import RouteHandler
from human_detection.utils.alerts import AlertManager, AlertCodes
from human_detection.config.settings import (
    HOST, PORT, DEBUG, CAMERA_INDEX, CAMERA_BACKEND
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class HumanDetectionApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.alert_manager = AlertManager()
        self.detector = HumanDetector()
        self.tracker = HumanTracker()
        self.route_handler = RouteHandler(self.app, self.alert_manager, self)
        self.reset_state()
    
    def reset_state(self):
        self.detection_mode = True
        self.was_tracking = False
        self.last_timestamp = "--:--:--"
    
    def force_redetection(self):
        self.detection_mode = True
        self.was_tracking = False
        self.tracker.tracker = None
        return True
    
    def process_frame(self, frame):
        disp = frame.copy()
        now = time.time()
        ts_str = time.strftime("%H:%M:%S", time.localtime(now))
        self.last_timestamp = ts_str
        draw_timestamp(disp, ts_str)
        
        if self.detection_mode:
            persons = self.detector.detect(frame)
            if persons:
                self.tracker.initialize(frame, persons)
                self.was_tracking = True
                self.detection_mode = False
                self.alert_manager.send_alert(AlertCodes.PERSON_DETECTED, "PERSON_DETECTED")
                draw_detection_boxes(disp, persons)
        elif self.tracker.tracker is not None:
            masks, has_mask = self.tracker.track(frame)
            if has_mask:
                bbox_coords = process_masks(masks, disp, frame)
                if self.tracker.check_stationary(bbox_coords, now):
                    self.alert_manager.send_alert(AlertCodes.STATIONARY_BEHAVIOR, 
                                                "STATIONARY BEHAVIOR DETECTED: analysis required")
            elif self.was_tracking:
                self.alert_manager.send_alert(AlertCodes.PERSON_LOST, "PERSON_LOST")
                self.reset_state()
        
        return disp
    
    def gen_frames(self):
        cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
        if not cap.isOpened():
            raise RuntimeError("camera is not opened")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            disp = self.process_frame(frame)
            
            ret2, buf = cv2.imencode('.jpg', disp)
            if not ret2:
                continue
                
            frame_bytes = buf.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   frame_bytes +
                   b'\r\n')
            time.sleep(0.01)
        
        cap.release()
    
    def run(self):
        self.app.run(host=HOST, port=PORT, debug=DEBUG)

if __name__ == '__main__':
    app = HumanDetectionApp()
    app.run() 