"""YuNet (phát hiện mặt + 5 landmark) và SFace (face embedding 128 chiều), đều chạy bằng OpenCV.

Hình học / liên tục của mặt chỉ là cổng chất lượng; nhận danh tính do embedding đảm nhiệm.
"""

import cv2
import numpy as np

from vn_av_data.common.runtime import require_file


class FaceTracker:
    def __init__(self, path, confidence=0.85, min_size=48, max_gap=2):
        require_file(path, "YuNet model; run 00_setup.py")
        self.detector = cv2.FaceDetectorYN.create(str(path), "", (320, 320), confidence)
        self.min_size = min_size
        # Hụt ≤ max_gap frame liên tiếp vẫn so liền mạch với mặt tốt gần nhất; dài hơn thì quên.
        self.max_gap = max_gap
        self.previous = None
        self.missed = 0

    def miss(self):
        self.missed += 1
        if self.missed > self.max_gap:
            self.previous = None

    def inspect(self, frame):
        """Đúng một mặt hợp lệ, liền mạch với mặt tốt gần nhất. Trả kèm dòng YuNet (15 số)."""
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        count = 0 if faces is None else len(faces)
        if count != 1:
            self.miss()
            return {"valid": False, "reason": "no_face" if count == 0 else "multiple_faces"}
        face = faces[0]
        x, y, w, h = face[:4].astype(float)
        points = face[4:14].reshape(5, 2)
        mouth = points[3:5]
        valid = (
            min(w, h) >= self.min_size
            and x >= 0
            and y >= 0
            and x + w <= width
            and y + h <= height
            and np.all(mouth[:, 0] > x)
            and np.all(mouth[:, 0] < x + w)
            and np.all(mouth[:, 1] > y + h * 0.4)
            and np.all(mouth[:, 1] < y + h)
            and np.linalg.norm(mouth[0] - mouth[1]) >= 0.12 * w
        )
        box = np.array([x, y, w, h])
        continuous = True
        if self.previous is not None:
            p = self.previous
            distance = np.linalg.norm(box[:2] + box[2:] / 2 - p[:2] - p[2:] / 2)
            continuous = bool(distance < max(w, h) * 0.6 and 0.5 < w / p[2] < 2)
        ok = bool(valid and continuous)
        if ok:
            self.previous, self.missed = box, 0
        else:
            self.miss()
        return {
            "valid": ok,
            "reason": "ok" if ok else "face_geometry_or_track_jump",
            "box": box.tolist(),
            "landmarks": points.astype(np.float32),
            "face": face.astype(np.float32),
        }


class FaceEmbedder:
    """SFace: căn mặt theo 5 landmark của YuNet → 112×112 → vector 128 chiều (chuẩn hoá L2)."""

    def __init__(self, path):
        require_file(path, "SFace model; run 00_setup.py")
        self.model = cv2.FaceRecognizerSF.create(str(path), "")

    def align(self, image, face):
        return self.model.alignCrop(image, face)

    def embed(self, aligned):
        feature = self.model.feature(aligned).reshape(-1).astype(np.float32)
        return feature / max(1e-8, float(np.linalg.norm(feature)))


def normalize(vector):
    vector = np.asarray(vector, np.float32)
    return vector / max(1e-8, float(np.linalg.norm(vector)))
