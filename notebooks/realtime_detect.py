import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO
from sam2.build_sam import build_sam2_object_tracker

if torch.cuda.is_available():
    idx = torch.cuda.current_device()
    print("Using GPU:", torch.cuda.get_device_name(idx))
else:
    print("Using CPU")


YOLO_MODEL_PATH     = os.path.abspath("checkpoints/yolov8n.pt")
SAM_CONFIG_PATH     = "./configs/samurai/sam2.1_hiera_b+.yaml"
SAM_CHECKPOINT_PATH = os.path.abspath("checkpoints/sam2.1_hiera_base_plus.pt")
DEVICE              = "cuda:0" if torch.cuda.is_available() else "cpu"
CAM_INDEX           = 1
REINIT_INTERVAL_S   = 5.0

def draw_bboxes(img, boxes):
    for (x1, y1), (x2, y2) in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

def main():
    yolo = YOLO(YOLO_MODEL_PATH)
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2)
    if not cap.isOpened():
        return

    last_reinit = time.time()
    sam = None
    tracked_boxes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        if sam is None or now - last_reinit >= REINIT_INTERVAL_S:
            results = yolo.predict(frame, classes=[0], device=DEVICE, verbose=False)
            persons = []
            for res in results:
                for b in res.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    persons.append([[x1, y1], [x2, y2]])
            if persons:
                sam = build_sam2_object_tracker(
                    num_objects=len(persons),
                    config_file=SAM_CONFIG_PATH,
                    ckpt_path=SAM_CHECKPOINT_PATH,
                    device=DEVICE,
                    verbose=False
                )
                sam.track_new_object(img=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), box=np.array(persons))
                tracked_boxes = persons.copy()
                last_reinit = now

        if sam is None:
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = sam.track_all_objects(img=img_rgb)
        masks = out.get('pred_masks')

        h, w = frame.shape[:2]
        overlay = frame.copy()
        curr_boxes = []

        if masks is not None:
            m_np = masks.cpu().numpy()
            for i in range(m_np.shape[0]):
                mask = (m_np[i, 0] > 0.5).astype(np.uint8)
                if mask.max() == 0:
                    continue
                mask = cv2.resize(mask, (w, h), cv2.INTER_NEAREST)
                overlay[mask > 0] = (0, 255, 0)
                ys, xs = np.where(mask > 0)
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                curr_boxes.append([[x1, y1], [x2, y2]])

        draw_bboxes(overlay, curr_boxes)
        disp = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        cv2.imshow("HumanSeg", disp)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
