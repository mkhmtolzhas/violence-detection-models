import os

from dotenv import load_dotenv


load_dotenv()


class Settings:
    def __init__(
        self,
        api_key: str = os.getenv("INFOBIP_API_KEY"),
        api_url: str = os.getenv("INFOBIP_API_URL"),
        alert_phone_number: str | None = os.getenv("ALERT_PHONE_NUMBER"),
        violence_confidence_threshold: float = float(
            os.getenv("VIOLENCE_CONFIDENCE_THRESHOLD", "0.80")
        ),
        violence_duration_seconds: float = float(
            os.getenv("VIOLENCE_DURATION_SECONDS", "0.7")
        ),
        voice_call_from: str = os.getenv("INFOBIP_FROM_NUMBER", "38515507799"),
        voice_call_text: str = os.getenv(
            "INFOBIP_VOICE_TEXT",
            "Violence detected. Please check immediately.",
        ),
    ):
        self.api_key = api_key
        self.api_url = api_url
        self.alert_phone_number = alert_phone_number
        self.violence_confidence_threshold = violence_confidence_threshold
        self.violence_duration_seconds = violence_duration_seconds
        self.voice_call_from = voice_call_from
        self.voice_call_text = voice_call_text


settings = Settings()
