import cv2
import sys
import json
import time
import math
import threading
import queue
from difflib import SequenceMatcher
from collections import deque, Counter
from openalpr import Alpr

class OpenALPRVideoAnalyzer:

    def __init__(
            self,
            frame_interval=5,
            min_confidence=0.3,
            track_threshold=60,
            scale_percent=50,
            num_workers=2,
            max_missed=10):

        self.frame_interval = frame_interval
        self.min_confidence = min_confidence
        self.track_threshold = track_threshold
        self.scale_percent = scale_percent
        self.num_workers = num_workers
        self.max_missed = max_missed

        self.tracks = {}
        self.next_id = 0

        self.lock = threading.Lock()

    @staticmethod
    def distance(p1, p2):
        return math.hypot(
            p1[0] - p2[0],
            p1[1] - p2[1]
        )

    def get_best_plate(self, track):
        if not track["plates"]:
            return None
        counter = Counter(track["plates"])
        return counter.most_common(1)[0][0]

    def plate_similarity(self, p1, p2):
        return SequenceMatcher(None, p1, p2).ratio()

    def get_direction(self, track):
        bboxes = list(track["bboxes"])
        
        if len(bboxes) < 5:
            return "неизвестно"

        y_centers = [(b[0] + b[2]) / 2 for b in bboxes]

        start_y = sum(y_centers[:3]) / 3
        end_y = sum(y_centers[-3:]) / 3

        movement_y = end_y - start_y

        pixel_threshold = 10

        if movement_y > pixel_threshold:
            return "к камере"
        elif movement_y < -pixel_threshold:
            return "от камеры"

        return "стационарно"

    def worker(self, frame_queue, result_queue, stop_event):
        try:
            alpr = Alpr(
                "eu",
                "/etc/openalpr/openalpr.conf",
                "/usr/share/openalpr/runtime_data"
            )

            if not alpr.is_loaded():
                print("Failed to load OpenALPR")
                return

            alpr.set_top_n(1)
        except Exception as e:
            print(f"Error initializing OpenALPR in thread: {e}")
            return

        while True:
            try:
                item = frame_queue.get(timeout=0.5)
            except queue.Empty:
                if stop_event.is_set():
                    break
                continue

            if item is None:
                frame_queue.task_done()
                break

            frame, frame_count, current_time = item

            _, encoded = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 85]
            )

            t0 = time.perf_counter()
            alpr_results = alpr.recognize_array(encoded.tobytes())
            alpr_time = time.perf_counter() - t0
            
            detections = []

            if alpr_results and "results" in alpr_results:
                for res in alpr_results["results"]:
                    if not res["candidates"]:
                        continue

                    best = res["candidates"][0]
                    confidence = best["confidence"] / 100.0

                    if confidence < self.min_confidence:
                        continue

                    plate = best["plate"]
                    coords = res.get("coordinates")

                    if not coords:
                        continue

                    xs = [p["x"] for p in coords]
                    ys = [p["y"] for p in coords]

                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                    detections.append({
                        "plate": plate,
                        "confidence": confidence,
                        "bbox": bbox,
                        "center": center,
                        "area": area
                    })

            result_queue.put((frame_count, current_time, detections, alpr_time))
            frame_queue.task_done()

    def analyze(self, source):
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print("Cannot open video")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25

        frame_queue = queue.Queue(maxsize=50)
        result_queue = queue.Queue()
        stop_event = threading.Event()
        workers = []

        for _ in range(self.num_workers):
            t = threading.Thread(
                target=self.worker,
                args=(frame_queue, result_queue, stop_event)
            )
            t.daemon = True
            t.start()
            workers.append(t)

        frame_count = 0
        results = []
        total_alpr_time = 0
        alpr_calls = 0
        start_processing = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                if frame_count % self.frame_interval:
                    continue

                if self.scale_percent < 100:
                    w = int(frame.shape[1] * self.scale_percent / 100)
                    h = int(frame.shape[0] * self.scale_percent / 100)
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

                current_time = frame_count / fps

                try:
                    frame_queue.put((frame, frame_count, current_time), timeout=0.05)
                except queue.Full:
                    continue

                raw_detections = []
                while not result_queue.empty():
                    raw_detections.append(result_queue.get())

                raw_detections.sort(key=lambda x: x[0])

                for fcnt, ctime, detections, alpr_time in raw_detections:
                    total_alpr_time += alpr_time
                    alpr_calls += 1

                    with self.lock:
                        for track in self.tracks.values():
                            track["missed"] += 1

                        for det in detections:
                            best_tid = None
                            best_dist = self.track_threshold

                            for tid, track in self.tracks.items():
                                d = self.distance(det["center"], track["last_center"])
                                if d < best_dist:
                                    best_dist = d
                                    best_tid = tid

                            if best_tid is not None:
                                tr = self.tracks[best_tid]
                                tr["last_center"] = det["center"]
                                tr["plates"].append(det["plate"])
                                tr["areas"].append(det["area"])
                                tr["bboxes"].append(det["bbox"])
                                tr["confidences"].append(det["confidence"])
                                tr["missed"] = 0
                            else:
                                tid = self.next_id
                                self.next_id += 1
                                self.tracks[tid] = {
                                    "last_center": det["center"],
                                    "plates": deque(maxlen=15),
                                    "areas": deque(maxlen=15),
                                    "bboxes": deque(maxlen=15),
                                    "confidences": deque(maxlen=15),
                                    "missed": 0
                                }

                        for tid in list(self.tracks.keys()):
                            if self.tracks[tid]["missed"] > self.max_missed:
                                del self.tracks[tid]

                    for tid, track in self.tracks.items():
                        if len(track["plates"]) < 3:
                            continue

                        plate = self.get_best_plate(track)
                        if not plate:
                            continue

                        avg_conf = sum(track["confidences"]) / len(track["confidences"])
                        direction = self.get_direction(track)

                        result = {
                            "time": ctime,
                            "plate": plate,
                            "score": avg_conf,
                            "direction": direction,
                            "bbox": track["bboxes"][-1]
                        }
                        results.append(result)

                        print(f"[{ctime:.2f}s] {plate:<12} {avg_conf:.3f} {direction}")

        finally:
            stop_event.set()

            while not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                    frame_queue.task_done()
                except queue.Empty:
                    break

            for _ in workers:
                try:
                    frame_queue.put(None, timeout=0.1)
                except queue.Full:
                    pass

            for t in workers:
                t.join(timeout=1.0)

            cap.release()


        if results:
            output_file = "results.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            print("No plates found")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    analyzer = OpenALPRVideoAnalyzer(
        frame_interval=5,
        scale_percent=50,
        num_workers=2,
        max_missed=10
    )

    analyzer.analyze(sys.argv[1])

