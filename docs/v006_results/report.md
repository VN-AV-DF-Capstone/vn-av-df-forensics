# Kết quả v006

Generator đã thấy khi train: iplap, musetalk_1_5, videoretalking, wav2lip_gan
Giọng đã thấy khi train: f5_vi, omnivoice, seedvc, vieneu

## Cách đọc

- Số seed mỗi model: 1. ± là độ lệch chuẩn giữa các seed (ddof 0); một seed thì không có ±.
- CI 95 % của cổng: bootstrap theo speaker (1000 lần, phân vị 2,5-97,5 %), trên seed đầu tiên: chỉ gồm biến động do mẫu người ở test, không gồm biến động do khởi tạo / thứ tự dữ liệu (chạy 1 seed).
- Không dò siêu tham số: baseline theo công thức của code gốc, M1 / M2 đặt trước; checkpoint và ngưỡng chọn trên validation; số lần mỗi model dùng test: `test-log.json`.
- Môi trường, commit, dữ liệu, cache: `provenance.json`; thời gian từng lần train: `<model>_seed<seed>/history.json`.

## AUC test (mean ± std theo seed)

| Model | ai | ai (cân theo người) | visual | audio | mismatch | ai (ref-norm) | định vị visual: ô cao nhất trong đoạn | định vị mismatch: AUC trong / ngoài đoạn |
|---|---|---|---|---|---|---|---|---|
| avh_align | 0.375 | 0.375 | 0.210 | 0.608 | 0.909 | 0.431 | 0.658 | 0.905 |
| avh_align_plus | 0.933 | 0.933 | - | - | 0.959 | 0.829 | - | 0.973 |
| avh_align_sup | 0.871 | 0.871 | 0.885 | 0.554 | 0.448 | 0.828 | 0.887 | 0.634 |
| bl_ens_av | 0.929 | 0.929 | 0.885 | 0.967 | - | 0.866 | - | - |
| lipinc_v2 | 0.630 | 0.630 | 0.696 | 0.466 | 0.435 | 0.629 | - | - |
| speechforensics | 0.490 | 0.490 | 0.456 | 0.513 | 0.668 | 0.522 | 0.320 | 0.814 |
| trihead_v5 | 0.971 | 0.971 | 0.990 | 0.965 | 0.971 | 0.946 | 0.970 | 0.985 |
| trihead_v5_no_pair | 0.973 | 0.973 | 0.992 | 0.966 | 0.973 | 0.949 | 0.977 | 0.985 |
| xlsr_frozen | 0.753 | 0.753 | - | 0.967 | - | 0.749 | - | - |

## Cổng (ΔAUC, bootstrap theo speaker, seed đầu tiên)

| So sánh | Tác vụ | Bản | ΔAUC | CI 95% | Speaker |
|---|---|---|---|---|---|
| M2 hơn BL-M/sup (cùng dữ liệu) (avh_align_plus − avh_align_sup) | mismatch | gốc | +0.512 | [+0.496; +0.527] | 31 |
| M2 hơn BL-M (avh_align_plus − avh_align) | mismatch | gốc | +0.050 | [+0.044; +0.056] | 31 |
| R1 M3-R hơn BL-Ens-AV (trihead_v5 − bl_ens_av) | ai | gốc | +0.042 | [+0.034; +0.052] | 31 |
| R2 M3-R hơn BL-Ens-AV, video nén 480p (trihead_v5 − bl_ens_av) | ai | c480 | +0.055 | [+0.043; +0.069] | 31 |
| R2b M3-R hơn BL-Ens-AV, video nén 360p (trihead_v5 − bl_ens_av) | ai | c360 | +0.059 | [+0.040; +0.078] | 31 |
| R3 M3-R hơn M2, video nén 480p (trihead_v5 − avh_align_plus) | ai | c480 | +0.051 | [+0.041; +0.060] | 31 |
| R6 cặp sạch ↔ nén giúp ở 480p (trihead_v5 − trihead_v5_no_pair) | ai | c480 | -0.001 | [-0.004; +0.002] | 31 |
| R8 M3-R không kém BL-A2, giọng chưa thấy, 480p (≥ −0,01) (trihead_v5 − xlsr_frozen) | audio | c480 | -0.001 | [-0.010; +0.007] | 31 |

## Điểm vận hành, tác vụ ai (seed đầu; ngưỡng = FPR 5 % trên real validation)

Precision ở tỉ lệ fake π ngoài thực tế = TPR·π / (TPR·π + FPR·(1 − π)) (Layton et al. 2024).

| Model | TPR | FPR test | precision π=50 % | π=10 % | π=1 % | báo nhầm / 1000 real |
|---|---|---|---|---|---|---|
| avh_align | 0.013 | 0.074 | 0.146 | 0.019 | 0.002 | 74 |
| avh_align_plus | 0.751 | 0.041 | 0.949 | 0.672 | 0.157 | 41 |
| avh_align_sup | 0.583 | 0.043 | 0.931 | 0.602 | 0.121 | 43 |
| bl_ens_av | 0.505 | 0.027 | 0.950 | 0.678 | 0.161 | 27 |
| lipinc_v2 | 0.133 | 0.036 | 0.787 | 0.291 | 0.036 | 36 |
| speechforensics | 0.017 | 0.039 | 0.299 | 0.045 | 0.004 | 39 |
| trihead_v5 | 0.816 | 0.010 | 0.988 | 0.898 | 0.444 | 10 |
| trihead_v5_no_pair | 0.836 | 0.017 | 0.980 | 0.842 | 0.327 | 17 |
| xlsr_frozen | 0.453 | 0.027 | 0.945 | 0.654 | 0.147 | 27 |

## Video nén lại (cfg["variants"]), tác vụ ai

Cùng mẫu test, nén lại video + tiếng (settings.VARIANTS). Ngưỡng FPR 5 % đặt trên real validation **sạch** (lúc dùng không biết chất lượng video) → FPR thực tế cho biết điểm real bị đẩy lên bao nhiêu khi nén; cột cuối: ngưỡng đặt trên validation cùng mức nén.

| Model | biến thể | AUC sạch (seed đầu) | AUC nén (mean ± std) | TPR (ngưỡng sạch) | FPR (ngưỡng sạch) | TPR (ngưỡng cùng mức) | FPR (ngưỡng cùng mức) |
|---|---|---|---|---|---|---|---|
| avh_align | c360 | 0.375 | 0.386 | 0.002 | 0.011 | 0.023 | 0.097 |
| avh_align | c480 | 0.375 | 0.380 | 0.007 | 0.039 | 0.013 | 0.060 |
| avh_align_plus | c360 | 0.933 | 0.855 | 0.609 | 0.097 | 0.400 | 0.024 |
| avh_align_plus | c480 | 0.933 | 0.909 | 0.680 | 0.043 | 0.650 | 0.033 |
| avh_align_sup | c360 | 0.871 | 0.777 | 0.389 | 0.075 | 0.267 | 0.036 |
| avh_align_sup | c480 | 0.871 | 0.841 | 0.463 | 0.048 | 0.419 | 0.034 |
| bl_ens_av | c360 | 0.929 | 0.849 | 0.427 | 0.033 | - | - |
| bl_ens_av | c480 | 0.929 | 0.904 | 0.458 | 0.027 | - | - |
| speechforensics | c360 | 0.490 | 0.467 | 0.094 | 0.115 | 0.007 | 0.024 |
| speechforensics | c480 | 0.490 | 0.461 | 0.023 | 0.050 | 0.017 | 0.037 |
| trihead_v5 | c360 | 0.971 | 0.907 | 0.775 | 0.104 | 0.370 | 0.015 |
| trihead_v5 | c480 | 0.971 | 0.960 | 0.798 | 0.017 | 0.700 | 0.004 |
| trihead_v5_no_pair | c360 | 0.973 | 0.910 | 0.774 | 0.094 | 0.374 | 0.008 |
| trihead_v5_no_pair | c480 | 0.973 | 0.961 | 0.812 | 0.023 | 0.717 | 0.007 |
| xlsr_frozen | c360 | 0.753 | 0.741 | 0.426 | 0.033 | 0.418 | 0.030 |
| xlsr_frozen | c480 | 0.753 | 0.752 | 0.453 | 0.027 | 0.466 | 0.034 |

## Định vị đoạn giả, tác vụ ai (seed đầu; chuẩn LAV-DF / AV-Deepfake1M)

| Model | AP@0.5 | AP@0.75 | AP@0.95 | AR@1 | AR@5 | AR@10 |
|---|---|---|---|---|---|---|
| avh_align | 0.006 | 0.002 | 0.000 | 0.003 | 0.064 | 0.142 |
| avh_align_plus | 0.024 | 0.011 | 0.000 | 0.011 | 0.174 | 0.291 |
| avh_align_sup | 0.001 | 0.000 | 0.000 | 0.002 | 0.021 | 0.042 |
| speechforensics | 0.001 | 0.000 | 0.000 | 0.001 | 0.022 | 0.065 |
| trihead_v5 | 0.023 | 0.016 | 0.000 | 0.012 | 0.191 | 0.318 |
| trihead_v5_no_pair | 0.022 | 0.015 | 0.001 | 0.011 | 0.173 | 0.313 |
| xlsr_frozen | 0.010 | 0.006 | 0.001 | 0.004 | 0.075 | 0.175 |

## Phép thử trễ: real test bị trượt tiếng k frame (AI = 0; seed đầu)

Báo nhầm ở ngưỡng FPR 5 % của real validation; AUC real trượt so với real gốc (0,5 = không bị lừa bởi độ trễ).

| Model | báo nhầm k=0 | k=2 | AUC k=2 | k=3 | AUC k=3 | k=4 | AUC k=4 |
|---|---|---|---|---|---|---|---|
| avh_align | 0.056 | 1.000 | 0.995 | 1.000 | 0.997 | 1.000 | 0.998 |
| avh_align_plus | 0.026 | 0.032 | 0.531 | 0.031 | 0.519 | 0.034 | 0.545 |
| avh_align_sup | 0.060 | 0.060 | 0.506 | 0.066 | 0.511 | 0.056 | 0.512 |
| speechforensics | 0.031 | 0.032 | 0.504 | 0.032 | 0.513 | 0.034 | 0.522 |
| trihead_v5 | 0.008 | 0.008 | 0.482 | 0.008 | 0.483 | 0.008 | 0.492 |
| trihead_v5_no_pair | 0.010 | 0.010 | 0.527 | 0.010 | 0.548 | 0.011 | 0.568 |

## Đầu loại thao tác: trihead_v5 (seed đầu, test)

Độ chính xác 0.819 trên 7368 mẫu; recall real 0.92, sham 0.95, voice 0.80, source 0.70, donor 0.73, both 0.88.

| ô thật \ dự đoán | real | sham | voice | source | donor | both |
|---|---|---|---|---|---|---|
| real | 571 | 25 | 7 | 12 | 4 | 1 |
| sham | 2 | 1161 | 51 | 0 | 5 | 3 |
| voice | 43 | 322 | 1480 | 2 | 4 | 9 |
| source | 195 | 6 | 3 | 869 | 156 | 9 |
| donor | 7 | 17 | 10 | 242 | 879 | 53 |
| both | 1 | 0 | 62 | 6 | 74 | 1077 |

## Đầu loại thao tác: trihead_v5_no_pair (seed đầu, test)

Độ chính xác 0.832 trên 7368 mẫu; recall real 0.93, sham 0.95, voice 0.81, source 0.75, donor 0.72, both 0.88.

| ô thật \ dự đoán | real | sham | voice | source | donor | both |
|---|---|---|---|---|---|---|
| real | 579 | 24 | 7 | 8 | 2 | 0 |
| sham | 1 | 1166 | 51 | 0 | 3 | 1 |
| voice | 26 | 312 | 1512 | 3 | 1 | 6 |
| source | 211 | 1 | 2 | 926 | 94 | 4 |
| donor | 9 | 17 | 6 | 252 | 873 | 51 |
| both | 2 | 2 | 45 | 4 | 90 | 1077 |

## Phân rã (seed đầu tiên của mỗi model, tác vụ ai)

- **avh_align**: cell/both 0.783, cell/donor 0.827, cell/source 0.694, cell/voice 0.957, generator/diff2lip 0.683, generator/iplap 0.771, generator/latentsync_1_5 0.482, generator/musetalk_1_5 0.864, generator/videoretalking 0.861, generator/wav2lip_gan 0.944, generator_seen/False 0.583, generator_seen/True 0.861, kind/full 0.831, kind/partial 0.832, voice/f5_inpaint 0.926, voice/f5_vi 0.850, voice/omnivoice 0.890, voice/seedvc 0.893, voice/vieneu 0.873, voice/vixtts 0.891, voice/voxcpm2 0.877, voice_seen/False 0.902, voice_seen/True 0.877, gender/F 0.839, gender/M 0.823
- **avh_align_plus**: cell/both 0.986, cell/donor 0.985, cell/source 0.935, cell/voice 0.951, generator/diff2lip 0.978, generator/iplap 0.990, generator/latentsync_1_5 0.859, generator/musetalk_1_5 0.994, generator/videoretalking 0.994, generator/wav2lip_gan 0.997, generator_seen/False 0.918, generator_seen/True 0.994, kind/full 0.980, kind/partial 0.949, voice/f5_inpaint 0.928, voice/f5_vi 0.971, voice/omnivoice 0.980, voice/seedvc 0.989, voice/vieneu 0.979, voice/vixtts 0.960, voice/voxcpm2 0.966, voice_seen/False 0.947, voice_seen/True 0.980, gender/F 0.958, gender/M 0.967
- **avh_align_sup**: cell/both 0.972, cell/donor 0.966, cell/source 0.900, cell/voice 0.748, generator/diff2lip 0.958, generator/iplap 0.981, generator/latentsync_1_5 0.799, generator/musetalk_1_5 0.973, generator/videoretalking 0.983, generator/wav2lip_gan 0.981, generator_seen/False 0.879, generator_seen/True 0.979, kind/full 0.935, kind/partial 0.834, voice/f5_inpaint 0.653, voice/f5_vi 0.888, voice/omnivoice 0.895, voice/seedvc 0.934, voice/vieneu 0.902, voice/vixtts 0.846, voice/voxcpm2 0.828, voice_seen/False 0.757, voice_seen/True 0.905, gender/F 0.878, gender/M 0.880
- **bl_ens_av**: cell/both 0.994, cell/donor 0.956, cell/source 0.888, cell/voice 0.938, generator/diff2lip 0.953, generator/iplap 0.975, generator/latentsync_1_5 0.830, generator/musetalk_1_5 0.967, generator/videoretalking 0.975, generator/wav2lip_gan 0.975, generator_seen/False 0.891, generator_seen/True 0.973, kind/full 0.960, kind/partial 0.930, voice/f5_inpaint 0.879, voice/f5_vi 0.992, voice/omnivoice 0.993, voice/seedvc 0.997, voice/vieneu 0.998, voice/vixtts 0.995, voice/voxcpm2 0.907, voice_seen/False 0.920, voice_seen/True 0.995, gender/F 0.944, gender/M 0.943
- **lipinc_v2**: cell/both 0.695, cell/donor 0.698, cell/source 0.693, cell/voice 0.500, generator/diff2lip 0.688, generator/iplap 0.710, generator/latentsync_1_5 0.653, generator/musetalk_1_5 0.707, generator/videoretalking 0.711, generator/wav2lip_gan 0.705, generator_seen/False 0.671, generator_seen/True 0.708, kind/full 0.790, kind/partial 0.502, voice/f5_inpaint 0.500, voice/f5_vi 0.561, voice/omnivoice 0.602, voice/seedvc 0.605, voice/vieneu 0.594, voice/vixtts 0.604, voice/voxcpm2 0.617, voice_seen/False 0.562, voice_seen/True 0.590, gender/F 0.618, gender/M 0.644
- **speechforensics**: cell/both 0.787, cell/donor 0.828, cell/source 0.756, cell/voice 0.818, generator/diff2lip 0.774, generator/iplap 0.835, generator/latentsync_1_5 0.654, generator/musetalk_1_5 0.822, generator/videoretalking 0.812, generator/wav2lip_gan 0.848, generator_seen/False 0.714, generator_seen/True 0.829, kind/full 0.896, kind/partial 0.723, voice/f5_inpaint 0.732, voice/f5_vi 0.816, voice/omnivoice 0.829, voice/seedvc 0.824, voice/vieneu 0.847, voice/vixtts 0.824, voice/voxcpm2 0.808, voice_seen/False 0.779, voice_seen/True 0.829, gender/F 0.817, gender/M 0.805
- **trihead_v5**: cell/both 0.998, cell/donor 0.995, cell/source 0.965, cell/voice 0.935, generator/diff2lip 0.986, generator/iplap 0.994, generator/latentsync_1_5 0.941, generator/musetalk_1_5 0.997, generator/videoretalking 0.998, generator/wav2lip_gan 0.998, generator_seen/False 0.963, generator_seen/True 0.997, kind/full 0.981, kind/partial 0.959, voice/f5_inpaint 0.883, voice/f5_vi 0.992, voice/omnivoice 0.995, voice/seedvc 0.998, voice/vieneu 0.998, voice/vixtts 0.997, voice/voxcpm2 0.896, voice_seen/False 0.919, voice_seen/True 0.996, gender/F 0.961, gender/M 0.976
- **trihead_v5_no_pair**: cell/both 0.998, cell/donor 0.995, cell/source 0.963, cell/voice 0.946, generator/diff2lip 0.987, generator/iplap 0.997, generator/latentsync_1_5 0.934, generator/musetalk_1_5 0.998, generator/videoretalking 0.998, generator/wav2lip_gan 0.999, generator_seen/False 0.961, generator_seen/True 0.998, kind/full 0.984, kind/partial 0.963, voice/f5_inpaint 0.897, voice/f5_vi 0.992, voice/omnivoice 0.997, voice/seedvc 0.998, voice/vieneu 0.998, voice/vixtts 0.998, voice/voxcpm2 0.919, voice_seen/False 0.932, voice_seen/True 0.996, gender/F 0.967, gender/M 0.977
- **xlsr_frozen**: cell/both 0.991, cell/donor 0.658, cell/source 0.503, cell/voice 0.970, generator/diff2lip 0.724, generator/iplap 0.725, generator/latentsync_1_5 0.719, generator/musetalk_1_5 0.709, generator/videoretalking 0.710, generator/wav2lip_gan 0.713, generator_seen/False 0.721, generator_seen/True 0.714, kind/full 0.752, kind/partial 0.842, voice/f5_inpaint 0.944, voice/f5_vi 0.992, voice/omnivoice 0.995, voice/seedvc 0.999, voice/vieneu 0.998, voice/vixtts 0.997, voice/voxcpm2 0.938, voice_seen/False 0.957, voice_seen/True 0.996, gender/F 0.802, gender/M 0.802
