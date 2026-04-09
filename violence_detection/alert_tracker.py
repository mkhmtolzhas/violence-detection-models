from violence_detection.call_service import call_to_number
from violence_detection.settings import settings


class ViolenceAlertTracker:
    def __init__(self):
        self.seconds_above_threshold = 0.0
        self.triggered = False

    def update(self, is_violence: bool, confidence: float, frame_duration: float) -> bool:
        if is_violence and confidence >= settings.violence_confidence_threshold:
            self.seconds_above_threshold += frame_duration
        else:
            self.seconds_above_threshold = 0.0

        return self.seconds_above_threshold >= settings.violence_duration_seconds

    def trigger(self):
        if self.triggered:
            return

        call_to_number(settings.alert_phone_number)
        self.triggered = True
