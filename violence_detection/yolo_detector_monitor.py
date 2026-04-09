import time

import cv2
from ultralytics import YOLO

from violence_detection.alert_tracker import ViolenceAlertTracker
from violence_detection.paths import MODELS_DIR, VIDEOS_DIR


WINDOW_NAME = "YOLOv8 Inference"
VIDEO_PATH = VIDEOS_DIR / "video3.mp4"
MODEL_PATH = MODELS_DIR / "yolo_small_weights.pt"
STATUS_BOX_TOP_LEFT = (10, 10)
STATUS_BOX_BOTTOM_RIGHT = (430, 95)
STATUS_BOX_COLOR = (0, 0, 255)


def build_model() -> YOLO:
    model = YOLO(str(MODEL_PATH))
    model.overrides["conf"] = 0.25
    model.overrides["iou"] = 0.45
    model.overrides["max_det"] = 1000
    return model


def draw_status_panel(frame, confidence: float, timer_seconds: float):
    cv2.rectangle(
        frame,
        STATUS_BOX_TOP_LEFT,
        STATUS_BOX_BOTTOM_RIGHT,
        STATUS_BOX_COLOR,
        -1,
    )
    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Violence timer: {timer_seconds:.1f}s",
        (20, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    model = build_model()
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1 / fps if fps and fps > 0 else 1 / 25
    frame_index = 0
    start_time = time.perf_counter()
    alert_tracker = ViolenceAlertTracker()

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model(frame)
            annotated_frame = results[0].plot()
            boxes = results[0].boxes
            max_confidence = 0.0

            if boxes is not None and boxes.conf is not None and len(boxes.conf) > 0:
                max_confidence = float(boxes.conf.max().item())

            should_alert = alert_tracker.update(
                is_violence=max_confidence > 0.0,
                confidence=max_confidence,
                frame_duration=frame_duration,
            )
            draw_status_panel(
                annotated_frame,
                confidence=max_confidence,
                timer_seconds=alert_tracker.seconds_above_threshold,
            )
            cv2.imshow(WINDOW_NAME, annotated_frame)

            frame_index += 1
            expected_time = frame_index * frame_duration
            elapsed_time = time.perf_counter() - start_time
            delay_ms = max(1, int((expected_time - elapsed_time) * 1000))

            if should_alert:
                alert_tracker.trigger()
                break

            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
