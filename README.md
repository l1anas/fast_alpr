<img width="2560" height="1440" alt="image" src="https://github.com/user-attachments/assets/4c420748-bed5-4439-976d-f169e0446b8d" /># FastALPR Video Stream Recognizer

Сервис распознавания автомобильных номерных знаков на видеофайлах, RTSP-потоках и USB-камерах с использованием **FastALPR** и **OpenCV**.

Проект использует OpenCV для получения кадров из различных источников видео и FastALPR для обнаружения и распознавания номерных знаков с помощью современных моделей глубокого обучения.

## Основные возможности

- распознавание номерных знаков на видео;
- работа с видеофайлами;
- работа с RTSP-потоками;
- работа с USB-камерами;
- выбор модели детекции;
- выбор OCR-модели;
- выбор устройства выполнения OCR (CPU/CUDA);
- настройка порога уверенности детектора;
- определение направления движения автомобиля;
- сохранение результатов в формате JSON.

---

# Установка

## Установка библиотеки

```bash
pip install opencv-python
```

FastALPR не устанавливает ONNX Runtime автоматически. Для выполнения инференса необходимо установить один из поддерживаемых backend'ов.

| Платформа / сценарий использования | Команда установки | Примечание |
|------------------------------------|-------------------|------------|
| **CPU (по умолчанию)** | `pip install fast-alpr[onnx]` | Кроссплатформенная версия (Linux, Windows, macOS) |
| **NVIDIA GPU (CUDA)** | `pip install fast-alpr[onnx-gpu]` | Использует ONNX Runtime с поддержкой CUDA |
| **Intel (OpenVINO)** | `pip install fast-alpr[onnx-openvino]` | Оптимизировано для процессоров Intel |
| **Windows (DirectML)** | `pip install fast-alpr[onnx-directml]` | Аппаратное ускорение через DirectML |
| **Qualcomm (QNN)** | `pip install fast-alpr[onnx-qnn]` | Для устройств Qualcomm |

---

# Используемые модели

По умолчанию используются следующие модели:

```python
self.alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v2-global-model"
)
```

## Модель детекции

По умолчанию используется

```
yolo-v9-t-384-license-plate-end2end
```

Доступные модели:

| Модель | Описание |
|---------|----------|
| `yolo-v9-s-608-license-plate-end2end` | Максимальная точность, самые высокие требования к вычислительным ресурсам |
| `yolo-v9-t-640-license-plate-end2end` | Высокая точность |
| `yolo-v9-t-512-license-plate-end2end` | Баланс между скоростью и качеством |
| `yolo-v9-t-416-license-plate-end2end` | Более высокая скорость при небольшом снижении точности |
| `yolo-v9-t-384-license-plate-end2end` | Используется по умолчанию |
| `yolo-v9-t-256-license-plate-end2end` | Самая быстрая модель |

---

## OCR-модель

По умолчанию используется

```
cct-xs-v2-global-model
```

Доступные OCR-модели:

| OCR модель | Размер | Архитектура | Производительность (FPS) |
|------------|---------|-------------|--------------------------|
| `cct-s-v1-global-model` | S | CCT | ≈1700 |
| `cct-xs-v1-global-model` | XS | CCT | ≈3090 |
| `cct-s-relu-v1-global-model` | S | CCT | ≈1760 |
| `cct-xs-relu-v1-global-model` | XS | CCT | ≈3200 |
| `cct-xs-v2-global-model` | XS | CCT | Используется по умолчанию |

---

# Использование других моделей

FastALPR позволяет использовать собственные модели детекции и OCR. Подробнее: https://ankandrew.github.io/fast-alpr/latest/custom_models/

# Конфигурация анализатора

Конфигурация выполняется при создании экземпляра класса.

```python
analyzer = FastALPRVideoAnalyzer(
    process_every_n_frames=10,
    output_dir="~/alpr",
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v2-global-model",
    detector_conf_thresh=0.4,
    ocr_device="auto"
)
```

## process_every_n_frames

Количество кадров, пропускаемых между анализом.

```python
process_every_n_frames=10
```

Каждый десятый кадр будет отправлен на распознавание.

Уменьшение значения увеличивает точность анализа, но снижает производительность.

---

## output_dir

Директория для сохранения JSON-файлов.

```python
output_dir="~/alpr"
```

Если директория отсутствует, она будет создана автоматически.

---

## detector_model

Модель детекции номерных знаков.

```python
detector_model="yolo-v9-t-384-license-plate-end2end"
```

Можно выбрать любую поддерживаемую модель.

---

## ocr_model

OCR-модель.

```python
ocr_model="cct-xs-v2-global-model"
```

Используется для распознавания символов номерного знака.

---

## detector_conf_thresh

Минимальная уверенность детектора.

```python
detector_conf_thresh=0.4
```

Все обнаружения с меньшей уверенностью будут отброшены.

---

## ocr_device

Устройство выполнения OCR.

```python
ocr_device="auto"
```

Поддерживаются:

- `auto`
- `cpu`
- `cuda`

---

# Запуск

Скрипт `fast_alpr.py` запускается из командной строки.

## Аргументы

| Аргумент | Обязательный | По умолчанию | Описание |
|----------|--------------|--------------|----------|
| `source` | Да | — | Путь к видеофайлу, RTSP URL или ID камеры |
| `--every` | Нет | `10` | Обрабатывать каждый N-й кадр |
| `--o` | Нет | `~/alpr` | Директория сохранения результатов |
| `--detector` | Нет | `yolo-v9-t-384-license-plate-end2end` | Модель детекции |
| `--ocr-model` | Нет | `cct-xs-v2-global-model` | OCR-модель |
| `--conf-thresh` | Нет | `0.4` | Минимальная уверенность детектора |
| `--ocr-device` | Нет | `auto` | Устройство выполнения OCR |

## Общий синтаксис

```bash
python3 fast_alpr.py <source> \
    [--every N] \
    [--o /path/to/results] \
    [--detector MODEL] \
    [--ocr-model MODEL] \
    [--conf-thresh VALUE] \
    [--ocr-device auto|cpu|cuda]
```

## Примеры

Видеофайл

```bash
python3 fast_alpr.py video.mp4
```

RTSP-поток

```bash
python3 fast_alpr.py rtsp://user:password@192.168.1.100:554/Streaming/Channels/101
```

USB-камера

```bash
python3 fast_alpr.py 0
```

Обработка каждого пятого кадра

```bash
python3 fast_alpr.py video.mp4 --every 5
```

Использование GPU

```bash
python3 fast_alpr.py video.mp4 --ocr-device cuda
```

Использование другой модели

```bash
python3 fast_alpr.py video.mp4 \
    --detector yolo-v9-s-608-license-plate-end2end \
    --ocr-model cct-s-v1-global-model
```

Сохранение результатов

```bash
python3 fast_alpr.py video.mp4 --o /home/user/results
```

---

# Поддерживаемые источники видео

Получение кадров выполняется с помощью OpenCV через интерфейс `cv2.VideoCapture`.

Поэтому поддерживаются любые источники видео, совместимые с OpenCV.

## Видеофайл

```bash
python3 fast_alpr.py video.mp4
```

---

## USB-камера

```bash
python3 fast_alpr.py 0
```

---

## RTSP-поток

```bash
python3 fast_alpr.py rtsp://user:password@192.168.1.100:554/Streaming/Channels/101
```

---

## HTTP/MJPEG поток

```bash
python3 fast_alpr.py http://192.168.1.100/mjpeg
```

---

# Формат результатов

Результаты автоматически сохраняются в формате JSON.

```json
[
  {
    "frame": 550,
    "result": {
      "plate": "AK64397",
      "detection_confidence": 0.8514284491539001,
      "ocr_confidence": 0.9718827207883199,
      "direction": "unknown",
      "bbox": [
        838,
        165,
        977,
        201
      ],
      "region": null,
      "region_confidence": null
    }
  }
]
```
