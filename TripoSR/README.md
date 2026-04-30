# TripoSR 2D to 3D (CPU)

This project provides a simple Gradio app that converts a 2D image into a 3D mesh using TripoSR.

## Features

- Image upload and one-click 3D reconstruction
- Background removal option
- 3D preview in browser (`.glb`)
- Downloadable mesh output (`.obj`)
- CPU-only runtime configuration

## Environment

- OS: Windows (Anaconda Prompt recommended)
- Python: 3.11
- Runtime: CPU only
- Conda environment name used in examples: `TripoSR`

## Configuration

Most project settings are defined near the top of `app.py`.

| Setting | Default | Description |
| --- | --- | --- |
| `DEVICE` | `cpu` | Runtime device. The current project is configured for CPU-only execution. |
| `MODEL_REPO` | `stabilityai/TripoSR` | Hugging Face model repository used by `TSR.from_pretrained`. |
| `TSR_REPO_URL` | `https://github.com/VAST-AI-Research/TripoSR.git` | TripoSR source repository cloned on first run if missing. |
| `TSR_LOCAL_DIR` | `third_party/TripoSR` | Local directory where the TripoSR source code is stored. |

The app also disables Gradio analytics at startup:

```python
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
```

By default, Gradio starts on port `7860`. If the port is already in use, change the launch call at the bottom of `app.py`, for example:

```python
app.launch(server_port=7861, inbrowser=True)
```

## First-run Downloads

The first run may take longer because the app prepares these files automatically:

- TripoSR source code: cloned into `third_party/TripoSR`
- TripoSR model weights: downloaded from Hugging Face cache for `stabilityai/TripoSR`
- Background removal assets: downloaded by `rembg` / `onnxruntime` when background removal is used

Make sure Git and internet access are available during the first run.

## Installation

1. Activate your conda environment.

```bash
conda activate TripoSR
```

1. Move to the project directory.

```bash
cd D:\Home\Projects\MachineLearning\TripoSR
```

1. Install dependencies.

```bash
pip install -r requirements.txt
```

1. (Optional) Install `torchmcubes` with helper script.

```bash
install_torchmcubes_conda.bat
```

## Run

```bash
python app.py
```

Then open:

- <http://127.0.0.1:7860>
- <http://localhost:7860>

## Notes

- On first run, `app.py` clones TripoSR source into `third_party/TripoSR` automatically if missing.
- The app is configured to run on CPU (`DEVICE = cpu`).
- `requirements.txt` uses the PyTorch CPU wheel index with `--extra-index-url https://download.pytorch.org/whl/cpu`.
- `torchmcubes` may require local build tools on Windows. Use `install_torchmcubes_conda.bat` from an activated conda environment if the normal install fails.

## Troubleshooting

- If `git` is not recognized, install Git for Windows and restart Anaconda Prompt.
- If the browser does not open automatically, manually open <http://127.0.0.1:7860>.
- If port `7860` is already in use, set another `server_port` in `app.py`.
- If model download fails, check internet access and Hugging Face availability, then run `python app.py` again.
