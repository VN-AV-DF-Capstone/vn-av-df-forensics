# Môi trường worker

Generator, TTS và AV-HuBERT có dependency xung đột nhau (Python, torch, fairseq, MMLab), nên mỗi cái chạy trong một môi trường riêng. Kernel chính chỉ gửi việc qua giao thức worker (`vn_av_df.worker`). Mọi môi trường dựng bằng Micromamba trên Kaggle/Linux, đặt ở `--env-root` (notebook: `WORKERS = /kaggle/temp/vn-av-df-workers`, ngoài Output 20 GB, mất khi hết phiên).

| Script | Môi trường | Dùng ở |
|---|---|---|
| [bootstrap_worker.py](bootstrap_worker.py) | `avhubert` ([avhubert.yml](avhubert.yml)), `musetalk` ([musetalk.yml](musetalk.yml)) | notebook prepare / generate |
| [worker_envs.py](worker_envs.py) | `gen_vrt`, `gen_iplap`, `gen_latentsync`, `gen_diff2lip`, `tts_vieneu`, `tts_f5`, `tts_omnivoice`, `tts_voxcpm`, `tts_xtts`, `vc_seedvc` | notebook generate |

Wav2Lip chạy trong Python chính (extra `generation`), không cần môi trường riêng.

```bash
# Từ project root, Linux x86_64, sau khi tải source/weights (generation/00_setup.py, training/00_setup.py).
python environments/bootstrap_worker.py avhubert --env-root /kaggle/temp/vn-av-df-workers
python environments/worker_envs.py tts_f5 --env-root /kaggle/temp/vn-av-df-workers
```

Script in đường dẫn Python của môi trường; notebook tự gán vào `cfg["envs"]`, `cfg["musetalk"]["python"]`, `cfg["encoders"]["avhubert"]["python"]`. Ở local, điền `ENVS`, `AVH_PYTHON`, `MUSETALK_PYTHON` trong `settings.py` (hoặc đặt `VN_AVH_PYTHON`, `VN_MUSETALK_PYTHON` trước khi import settings).

- **Receipt:** chỉ đánh dấu ready sau khi cài xong và probe import/CUDA đạt. Chạy lại cùng cấu hình thì cài tiếp phần dở; môi trường đã ready không cài lại. Đổi công thức → môi trường mới.
- **Log:** `<env-root>/.kaggle-tools/<tên>/setup.log`. Notebook in 80 dòng cuối khi lỗi.
- **Trạng thái:** `bootstrap_worker.py` đã chạy được trên Kaggle ở v002 (AV-HuBERT Base, MuseTalk). Công thức trong `worker_envs.py` viết theo README / requirements của từng upstream, **chưa chạy thử trên Kaggle**.
- Script không chạy trên Windows, tránh cài nhầm môi trường Kaggle lên máy local.
