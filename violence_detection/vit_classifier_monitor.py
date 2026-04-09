import os
import time

os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")

import cv2
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import (
    TimmWrapperConfig,
    TimmWrapperForImageClassification,
    ViTImageProcessor,
)

from violence_detection.alert_tracker import ViolenceAlertTracker
from violence_detection.paths import VIDEOS_DIR


WINDOW_NAME = "Violence Classification"
VIDEO_PATH = VIDEOS_DIR / "video3.mp4"
MODEL_ID = "jaranohaal/vit-base-violence-detection"
MODEL_ARCHITECTURE = "vit_base_patch16_224"
INFER_EVERY_N_FRAMES = 5
LABELS = {
    0: "class_0",
    1: "class_1",
}
VIOLENCE_CLASS_IDX = 1
STATUS_PANEL_TOP_LEFT = (10, 10)
STATUS_PANEL_BOTTOM_RIGHT = (430, 130)


def build_model():
    model_dir = snapshot_download(MODEL_ID, local_files_only=True)
    config = TimmWrapperConfig(
        architecture=MODEL_ARCHITECTURE,
        num_labels=2,
    )
    model = TimmWrapperForImageClassification.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True,
    )
    image_processor = ViTImageProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, image_processor, device


def draw_status_panel(frame, label: str, confidence: float, timer_seconds: float, color):
    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), color, 8)
    cv2.rectangle(frame, STATUS_PANEL_TOP_LEFT, STATUS_PANEL_BOTTOM_RIGHT, color, -1)
    cv2.putText(
        frame,
        f"Predicted: {label}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (20, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Violence timer: {timer_seconds:.1f}s",
        (20, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    model, image_processor, device = build_model()
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1 / fps if fps and fps > 0 else 1 / 25
    frame_index = 0
    start_time = time.perf_counter()
    last_predicted_class_idx = 0
    last_confidence = 0.0
    alert_tracker = ViolenceAlertTracker()

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_index % INFER_EVERY_N_FRAMES == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_frame)
                inputs = image_processor(images=image, return_tensors="pt")
                inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                    last_predicted_class_idx = probabilities.argmax(dim=-1).item()
                    last_confidence = probabilities[0][last_predicted_class_idx].item()

            label = LABELS.get(last_predicted_class_idx, f"class_{last_predicted_class_idx}")
            is_violence = last_predicted_class_idx == VIOLENCE_CLASS_IDX
            color = (0, 0, 255) if is_violence else (0, 200, 0)
            should_alert = alert_tracker.update(
                is_violence=is_violence,
                confidence=last_confidence,
                frame_duration=frame_duration,
            )
            draw_status_panel(
                frame,
                label=label,
                confidence=last_confidence,
                timer_seconds=alert_tracker.seconds_above_threshold,
                color=color,
            )

            frame_index += 1
            expected_time = frame_index * frame_duration
            elapsed_time = time.perf_counter() - start_time
            delay_ms = max(1, int((expected_time - elapsed_time) * 1000))

            cv2.imshow(WINDOW_NAME, frame)

            if should_alert:
                alert_tracker.trigger()
                break

            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
