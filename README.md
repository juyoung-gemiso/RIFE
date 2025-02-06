# Frame Interpolation

## Purpose

- 방송에서 사용하는 interlaced 영상에 interpolation을 적용하여 더 나은 품질의 interlaced 영상을 출력
- 유료 프로그램인 Topaz와 ffmpeg을 대체할 수 있는 것이 목표

## Getting Started

### Prerequisites

- create conda env
    ```bash
    conda env create -n rife python=3.10
    ```

- install pytorch
    ```bash
    pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
    ```

### Installation

1. Clone the repo
2. Install requirements

    ```bash
    pip install -r requirements.txt
    ```
## Usage

```bash
python inference_v6.py \
--video=Z:\\AI_workspace\\video_interpolation_backup\\soccer_25fps_1m30s.mxf \
--output_base_path=Z:\\AI_workspace\\video_interpolation_backup \
--fps=29.97 \
--ext=mxf
```

- video: video file path(*.mxf)
- output_base_path: 추출된 프레임과 출력 영상이 저장되는 경로
- fps: 출력 영상의 framerate
- ext: 출력 영상의 확장자

## Demo

```bash
python -m demo.demo
```

## Development Environment

| 구분 | 정보 |
| --- | --- |
| OS | Windows 10 |
| Language | Python3.10.16 |
| Framework | Pytorch2.1.1+cu118 |
| Linux Remote IP | 192.168.1.202 |
| Docker Container Name | tr_ocr |
| GPU | NVIDIA GeForce RTX 3090 - 24576MiB |
| CUDA version | 12.2 |
