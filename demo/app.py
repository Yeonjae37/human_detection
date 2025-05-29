import os
import time
import cv2
import numpy as np
import torch
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
from sam2.build_sam import build_sam2_object_tracker
from alerts import AlertManager, AlertCodes

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # macOS나 일부 환경에서 Intel MKL 중복 로딩 오류를 피하기 위한 환경 변수 설정

last_timestamp = "--:--:--"

class HumanDetectionApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.alert_manager = AlertManager()
        self.setup_config()
        self.setup_models()
        self.setup_routes()
        self.reset_state()

    def setup_config(self):
        self.YOLO_MODEL_PATH = os.path.abspath("checkpoints/yolov8n.pt")
        self.SAM_CONFIG_PATH = "./configs/samurai/sam2.1_hiera_b+.yaml"
        self.SAM_CHECKPOINT_PATH = os.path.abspath("checkpoints/sam2.1_hiera_base_plus.pt")
        self.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    def setup_models(self):
        self.yolo = YOLO(self.YOLO_MODEL_PATH)
        self.sam = None # SAM 추적기는 나중에 첫 검출 시에 생성

    def reset_state(self): 
        self.detection_mode = True # YOLO 검출 모드
        self.was_tracking = False # SAM2 추적 실행 여부 플래그
        self.stationary_timer_start = None # 정지 감지용 타이머
        self.last_center = None # 이전 중심점

    def setup_routes(self):
        self.app.route('/')(self.index)
        self.app.route('/video_feed')(self.video_feed)
        self.app.route('/alerts')(self.alerts)
        self.app.route('/redetect', methods=['POST'])(self.redetect)
        self.app.route('/timestamp')(self.timestamp)

    def timestamp(self):
        return jsonify({'timestamp': last_timestamp}) # last_timestamp를 JSON으로 반환
    
    def index(self): 
        return render_template('index.html') # index.html 템플릿 렌더링

    def alerts(self):
        def event_stream():
            self.alert_manager.send_alert(AlertCodes.SYSTEM_STARTED, "SYSTEM_STARTED: waiting for human")
            while True:
                data = self.alert_manager.get_next_alert()
                if data:
                    yield f"data: {data}\n\n"
                else:
                    yield "data: \n\n"
        return Response(event_stream(), mimetype='text/event-stream')

    def redetect(self):
        self.reset_state()
        return jsonify({'success': True})

    def process_detection(self, frame):
        persons = []
        results = self.yolo.predict(frame, classes=[0], device=self.DEVICE, verbose=False)
        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                persons.append([[x1, y1], [x2, y2]])

        if persons:
            self.sam = build_sam2_object_tracker(
                num_objects=len(persons),
                config_file=self.SAM_CONFIG_PATH,
                ckpt_path=self.SAM_CHECKPOINT_PATH,
                device=self.DEVICE,
                verbose=False
            )
            self.sam.track_new_object(
                img=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                box=np.array(persons)
            )
            self.was_tracking = True
            self.detection_mode = False
            self.alert_manager.send_alert(AlertCodes.PERSON_DETECTED, "PERSON_DETECTED")
            return True
        return False

    def process_masks(self, m_np, disp, frame):
        h, w = disp.shape[:2]
        bbox_coords = None # 바운딩박스 좌표 저장용 변수
        for i in range(m_np.shape[0]): # m_np : SAM2가 반환한 마스크 
            mask = (m_np[i,0] > 0.5).astype(np.uint8)  # 0.5 임계값 이상인 픽셀을 True, 나머지는 False
            if mask.sum() == 0: # 1로 표시된 픽셀 개수가 0이면 해당 마스크에 객체 없음
                continue
            mask = cv2.resize(mask, (w, h), cv2.INTER_NEAREST) 
            disp[mask>0] = (0,255,0) # mask > 0인 픽셀 위치를 인덱싱하여 초록색 덮어씌움
            ys, xs = np.where(mask>0)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            bbox_coords = (x1, y1, x2, y2)
            cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,255), 2) # 바운딩박스 그리기
        disp = cv2.addWeighted(disp, 0.5, frame, 0.5, 0) # 원본 프레임과 마스크 결과를 50%씩 섞어서 표시
        return bbox_coords # 바운딩박스 좌표 반환

    def check_stationary_behavior(self, bbox_coords):
        if bbox_coords is None: # 바운딩박스 좌표가 없으면 함수 종료
            return
            
        # (x1, y1) : 좌상단
        # (x2, y2) : 우하단
        # cx, cy : 중심점
        x1, y1, x2, y2 = bbox_coords
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        now = time.time()

        if self.last_center is None: # 메서드 처음 호출 시
            self.last_center = (cx, cy) # 현재 중심점 저장
            self.stationary_timer_start = now # 정지 감지 타이머 시작 시각 기록
        else:
            dist = np.hypot(cx - self.last_center[0], cy - self.last_center[1]) # 이동 거리 계산 : 이전 중심과 현재 중심 사이의 유클리드 거리 계산
            if dist < 5: # 5px 미만이면 정지 상태로 간주
                if self.stationary_timer_start and now - self.stationary_timer_start >= 3.0: # 3초 이상 정지 상태 유지 시 경고 발생
                    self.alert_manager.send_alert(AlertCodes.STATIONARY_BEHAVIOR, "STATIONARY BEHAVIOR DETECTED: analysis required")
                    self.stationary_timer_start = None # 정지 타이머 초기화 -> 중복 알림 방지
            else:
                self.last_center = (cx, cy) # 이동 시 중심점 업데이트
                self.stationary_timer_start = now # 정지 타이머도 현재 시각으로 재설정

    def process_tracking(self, frame, disp):
        out = self.sam.track_all_objects(img=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # out에는 pred_masks, pred_boxes, pred_scores, pred_labels 키가 있음
        masks = out.get("pred_masks") # 마스크 확률맵(Tensor) 꺼냄
        has_mask = False # 마스크 감지 여부 플래그

        if masks is not None: # masks가 None이 아니면 마스크 감지 플래그 설정
            m_np = masks.cpu().numpy()
            for i in range(m_np.shape[0]): # 각 마스크 레이어를 살펴봄
                if (m_np[i,0] > 0.5).sum() > 0: # 0.5 이상 픽셀을 객체로 간주해 이진화
                    has_mask = True # 마스크 감지 플래그 설정
                    break # 유효 마스크가 확인되면(첫 번째 객체만 처리)

            if has_mask: # 마스크 감지 플래그가 True이면 바운딩박스 좌표 처리
                bbox_coords = self.process_masks(m_np, disp, frame) # 바운딩박스 좌표 반환
                self.check_stationary_behavior(bbox_coords) # 정지 행동 감지 처리

        if self.was_tracking and not has_mask: # 추적 중인 상황에서 더 이상 마스크가 검출되지 않으면
            self.alert_manager.send_alert(AlertCodes.PERSON_LOST, "PERSON_LOST")
            self.reset_state()

        return has_mask

    def gen_frames(self):
        global last_timestamp
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError("camera is not opened")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            h, w = frame.shape[:2]
            now = time.time()
            ts_str = time.strftime("%H:%M:%S", time.localtime(now))
            last_timestamp = ts_str  

            text_size, _ = cv2.getTextSize(ts_str, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            text_w, text_h = text_size

            disp = frame.copy()
            cv2.putText(
                disp,
                ts_str,
                (w - text_w - 10, text_h + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            if self.detection_mode: 
                detected = self.process_detection(frame)
                if detected:
                    for r in self.yolo.predict(frame, classes=[0], device=self.DEVICE, verbose=False):
                        for b in r.boxes:
                            x1, y1, x2, y2 = map(int, b.xyxy[0])
                            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
            elif self.sam is not None:
                self.process_tracking(frame, disp)

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

    def video_feed(self):
        return Response(self.gen_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    def run(self):
        self.app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    app = HumanDetectionApp()
    app.run()
