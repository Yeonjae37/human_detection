import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, render_template, Response
import cv2, time, numpy as np, torch
from ultralytics import YOLO
from sam2.build_sam import build_sam2_object_tracker

app = Flask(__name__)

# ── 모델 초기화 ──────────────────────────────────────
YOLO_MODEL_PATH     = os.path.abspath("checkpoints/yolov8n.pt")
SAM_CONFIG_PATH     = "./configs/samurai/sam2.1_hiera_b+.yaml"
SAM_CHECKPOINT_PATH = os.path.abspath("checkpoints/sam2.1_hiera_base_plus.pt")
DEVICE              = "cuda:0" if torch.cuda.is_available() else "cpu"
REINIT_INTERVAL_S   = 1.0

yolo = YOLO(YOLO_MODEL_PATH)
sam = None
last_reinit = 0

# ── 영상 스트림 생성기 ─────────────────────────────────
def gen_frames():
    global sam, last_reinit
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        raise RuntimeError("camera is not opened")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError("camera is not opened")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Error] failed to read frame")
            break

        h, w = frame.shape[:2]
        now = time.time()

        # 1초마다 YOLO로 재검출 → SAM 재초기화
        if sam is None or now - last_reinit >= REINIT_INTERVAL_S:
            persons = []
            results = yolo.predict(frame, classes=[0], device=DEVICE, verbose=False)
            for res in results:
                for box in res.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    persons.append([[x1, y1], [x2, y2]])
            if persons:
                sam = build_sam2_object_tracker(
                    num_objects=len(persons),
                    config_file=SAM_CONFIG_PATH,
                    ckpt_path=SAM_CHECKPOINT_PATH,
                    device=DEVICE,
                    verbose=False
                )
                sam.track_new_object(
                    img=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    box=np.array(persons)
                )
                last_reinit = now

        # SAM2 트래킹 & 오버레이
        disp = frame.copy()
        if sam is not None:
            out = sam.track_all_objects(img=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            masks = out.get("pred_masks")
            if masks is not None:
                m_np = masks.cpu().numpy()
                for i in range(m_np.shape[0]):
                    mask = (m_np[i,0] > 0.5).astype(np.uint8)
                    if mask.max() == 0: continue
                    mask = cv2.resize(mask, (w, h), cv2.INTER_NEAREST)
                    disp[mask>0] = (0,255,0)
                    ys, xs = np.where(mask>0)
                    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0,255,255), 2)
            disp = cv2.addWeighted(disp, 0.5, frame, 0.5, 0)

        # JPEG 인코딩 후 스트림
        ret2, buffer = cv2.imencode('.jpg', disp)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.01)
    
    cap.release()

# ── Flask 라우팅 ─────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 호스트, 포트, 디버그 모드는 필요에 따라 수정하세요
    app.run(host='0.0.0.0', port=5000, debug=True)
