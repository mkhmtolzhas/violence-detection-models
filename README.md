# Violence Monitoring

Monitoring project for video-based violence detection with two pipelines:
- YOLO-based detection
- ViT-based frame classification

## Structure

```text
.
├── assets/
│   ├── models/
│   │   └── yolo_small_weights.pt
│   └── videos/
│       └── video3.mp4
├── violence_detection/
│   ├── alert_tracker.py
│   ├── call_service.py
│   ├── paths.py
│   ├── settings.py
│   ├── vit_classifier_monitor.py
│   └── yolo_detector_monitor.py
├── run_vit_monitor.py
├── run_yolo_monitor.py
├── pyproject.toml
└── README.md
```

## Configuration

Copy `.env.example` to `.env` and fill in your real credentials:

```env
INFOBIP_API_KEY=your_key
INFOBIP_API_URL=your_url
ALERT_PHONE_NUMBER=77767301903
VIOLENCE_CONFIDENCE_THRESHOLD=0.80
VIOLENCE_DURATION_SECONDS=0.7
INFOBIP_FROM_NUMBER=38515507799
INFOBIP_VOICE_TEXT=Violence detected. Please check immediately.
```

## Install uv

Official docs:
- https://docs.astral.sh/uv/getting-started/installation/

Official download/install page:
- https://astral.sh/uv

macOS and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Check that `uv` is installed:

```bash
uv --version
```

## Setup With uv

Install dependencies and create the virtual environment from the project root:

```bash
uv sync
```

If you want to activate the environment manually:

```bash
source .venv/bin/activate
```

You can also run scripts without manual activation:

```bash
uv run python run_yolo_monitor.py
uv run python run_vit_monitor.py
```

## Run

From the project root:

```bash
python run_yolo_monitor.py
python run_vit_monitor.py
```

If the package is installed, these also work:

```bash
run-yolo-monitor
run-vit-monitor
```

## Notes

- The alert is triggered when violence is detected continuously for about `0.7` seconds with confidence above `0.80`.
- Model weights and sample videos are kept in `assets/`.
- Large binaries and environment files are ignored by `.gitignore`.
