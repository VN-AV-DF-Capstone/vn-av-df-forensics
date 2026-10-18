"""Cấu hình mặc định cho các file .py ở local (generation/, training/).

Data/review: chọn PART trong data_settings.py. Trên Kaggle: chỉnh các cell cấu hình trong
notebook, không cần sửa file này rồi upload lại. config() lấy giá trị hiện tại; trong notebook
chỉ gọi nó sau khi chọn đường dẫn/model. RUN_NAME là folder kết quả, không phải tên dataset.
"""

import os
import sys
from pathlib import Path

from data_settings import DATASET_NAME, PART, part_name

ROOT = Path(__file__).resolve().parent
CLEAN_DATASET = ROOT / "data_pipeline/exports" / DATASET_NAME / part_name()
GENERATION_NAME = part_name()
DATASET = ROOT / "datasets" / DATASET_NAME
# "all" = mọi part đã tải vào DATASET; hoặc ["vn-av-df-data-part1", ...].
DATASET_PARTS = [part_name()]
RUN_NAME = "train_part1_v003"  # Output: runs/<RUN_NAME>/<model>_seed<seed>/.
DEVICE = "cuda"  # Kaggle GPU. Use "cpu" for a local CPU-only environment.

# Generator và hệ giọng: nhóm đầu dùng cho train/validation/test, nhóm sau chỉ ở test.
TRAIN_GENERATORS = ["wav2lip_gan", "musetalk_1_5", "iplap", "videoretalking"]
TEST_ONLY_GENERATORS = ["latentsync_1_5", "diff2lip"]
TRAIN_VOICES = ["vieneu", "f5_vi", "omnivoice", "seedvc"]
TEST_ONLY_VOICES = ["voxcpm2", "vixtts"]
# 6 baseline + 3 đề xuất; ablation thêm khi cần. Xem docs/v003_model_research.md.
MODELS = [
    "speechforensics",
    "avh_align",
    "avh_align_sup",
    "biolip",
    "av_lipsync_plus",
    "xlsr_aasist",
    "avcv",
    "mtn",
    "trihead",
]
ABLATIONS = [
    "avcv_no2d",
    "avcv_cos",
    "mtn_no_npr",
    "mtn_no_temporal",
    "mtn_no_kin",
    "mtn_no_aug",
    "trihead_no_ref",
    "trihead_no_audio",
    "trihead_no_trace",
    "trihead_no_sync",
]
SEEDS = [42, 43, 44]
# Python của từng môi trường worker (environments/worker_envs.py in ra đường dẫn sau khi cài).
ENVS = {}
AVH_PYTHON = os.environ.get(
    "VN_AVH_PYTHON",
    str(ROOT / ".venv-avhubert" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")),
)
MUSETALK_PYTHON = os.environ.get(
    "VN_MUSETALK_PYTHON",
    str(ROOT / ".venv-musetalk" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")),
)


def config():
    """Cấu hình dùng chung cho Run File và notebook; không hardcode thuật toán trong notebook."""
    train = {"generators": TRAIN_GENERATORS, "voices": TRAIN_VOICES}
    test = {
        "generators": TRAIN_GENERATORS + TEST_ONLY_GENERATORS,
        "voices": TRAIN_VOICES + TEST_ONLY_VOICES,
    }
    cfg = {
        "seed": 42,
        "seeds": SEEDS,
        "device": DEVICE,
        "root": str(ROOT),
        "clean_dataset": str(CLEAN_DATASET),
        "generated_dataset": str(DATASET / GENERATION_NAME),
        "data_part": PART,
        "dataset": str(DATASET),
        "dataset_parts": DATASET_PARTS,
        "plan": str(ROOT / "generation/plans" / f"{GENERATION_NAME}.json"),
        "cache": str(ROOT / "cache" / DATASET_NAME),
        "runs": str(ROOT / "runs" / RUN_NAME),
        "envs": dict(ENVS),
        "models": MODELS,
        "generation": {
            "generators": {
                "train": train["generators"],
                "validation": train["generators"],
                "test": test["generators"],
            },
            "voices": {
                "train": train["voices"],
                "validation": train["voices"],
                "test": test["voices"],
            },
            "inpaint": {"test": "f5_inpaint"},  # track C (sửa trực tiếp) chỉ ở test
            # Mỗi clip chỉ chạy một nửa số generator (rút cân bằng, xoay ô); giọng giữ nguyên.
            # 20 clip / người × một nửa generator: cùng GPU với 10 clip × mọi generator nhưng gấp
            # đôi số clip real làm nền (v003_model_research.md §10).
            "generators_per_clip": {"train": 2, "validation": 2, "test": 3},
            "partial_range_s": [0.3, 3.0],  # gồm cả đoạn sửa 1 từ (LAV-DF: trung bình 0,65 s)
            "edge_margin_s": 0.5,
            "blend_probability": 0.5,
            "blend_frames": [2, 6],
            "max_side": 1280,  # giữ nguyên 720p (1280×720) khi sinh
            "crf": 18,
            "asr_model": "vinai/PhoWhisper-small",
            "voice_models": str(ROOT / "weights/voice"),
            "iplap_checkpoints": None,  # thư mục có 2 file .pth của IP-LAP (tải tay từ OneDrive)
            "latentsync_config": "configs/unet/stage2.yaml",
            "references": 3,  # số clip real tham chiếu mỗi người (ref-norm, TriHead-Ref)
        },
        "wav2lip": {
            "repo": str(ROOT / "external/Wav2Lip"),
            "checkpoint": str(ROOT / "weights/wav2lip/Wav2Lip-SD-GAN.pt"),
            "device": DEVICE,
            "python": sys.executable,
        },
        "musetalk": {
            "repo": str(ROOT / "external/MuseTalk"),
            "python": MUSETALK_PYTHON,
            "device": DEVICE,
            "batch_size": 4,
            "seed": 42,
        },
        "training": {
            "epochs": 30,
            "patience": 5,
            "lr": 3e-4,
            "avh_align_lr": 1e-5,
            "weight_decay": 1e-4,
            "hidden": 128,
            "dropout": 0.1,
            "batch_size": 16,
            "eval_batch_size": 8,
            "workers": 2,
            # Dịch tiếng so với hình k frame ∈ [−1, +2] khi train (dương = tiếng trễ); xem
            # docs/v003_model_research.md §9 (BT.1359, sai số đo, điểm 0 generator).
            "audio_lag_aug": [-1, 2],
        },
        "bootstrap_replicates": 1000,
    }
    cfg["encoders"] = {
        "avhubert": {
            "kind": "avhubert",
            "repo": str(ROOT / "external/av_hubert"),
            # AVH-Align dùng AV-HuBERT Large (VoxCeleb2 + LRS3, fine-tune VSR 433 h).
            "checkpoint": str(ROOT / "weights/avhubert/large_vox_433h.pt"),
            "checkpoint_url": "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/large_vox_433h.pt",
            "allow_finetuned": True,
            "landmarks": str(ROOT / "weights/avhubert/shape_predictor_68_face_landmarks.dat"),
            "mean_face": str(ROOT / "weights/avhubert/20words_mean_face.npy"),
            "mean_face_url": "https://raw.githubusercontent.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/master/preprocessing/20words_mean_face.npy",
            "device": DEVICE,
            "python": AVH_PYTHON,
            "chunk_frames": 200,
            "overlap_frames": 50,
            "output_stride": 5,
            "max_duration": 60,
            "max_side": 640,
            "workers_per_device": 2,
        },
        "xlsr": {
            "kind": "xlsr",
            "model": str(ROOT / "weights/wav2vec2-xls-r-300m"),
            "hf_id": "facebook/wav2vec2-xls-r-300m",
            "layer_low": 6,
            "layer_high": 12,
            "device": DEVICE,
        },
        "mouth": {"kind": "mouth", "crop": 96, "box_max_side": 640},
    }
    return cfg


def bootstrap():
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "data_pipeline/src")]
