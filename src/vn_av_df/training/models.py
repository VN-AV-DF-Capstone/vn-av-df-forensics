"""Model của v003: 6 baseline + 3 kiến trúc đề xuất, cùng một giao diện.

Mỗi model:
- needs: cache cần nạp ({"av", "ssl", "mouth"});
- heads: {tên đầu ra: tác vụ} - tác vụ ∈ ai | visual | audio | mismatch; forward trả logit /
  điểm theo frame [B, T] cho từng đầu (điểm cao = nghi ngờ);
- pool: cách gộp điểm frame thành điểm video ("topk" trên xác suất, "lse", "mean");
- loss(batch, out): mặc định BCE theo frame + BCE video (top-k) cho từng đầu có nhãn.

Baseline (đúng kiến trúc trong paper; khác biệt ghi trong docstring):
- speechforensics (NeurIPS 2024): không train; cos(a_t, v_{t+δ}) với δ cố định tốt nhất ±15.
- avh_align (CVPR 2025): MLP 2D→512→256→128→1 (LN + ReLU), InfoNCE trên ±15 frame, chỉ real;
  điểm frame −Φ_ii, gộp log-sum-exp. avh_align_sup: cùng mạng, BCE trên σ(LSE).
- biolip (2026): động học landmark môi (y; 4 bậc), cửa sổ 25 frame bước 5, Conv1D 3 khối +
  thống kê + tỉ lệ vùng. Khác paper: 20 điểm môi dlib thay vì 64 điểm MediaPipe.
- av_lipsync_plus (≈ B-AVH): AV-HuBERT a‖v → TCN giãn 1/2/4, có giám sát nhãn AI.
- xlsr_aasist: XLS-R → AASIST-lite (self-attention theo thời gian thay cho HS-GAL đầy đủ).
Đề xuất (docs/v003_model_research.md §5): avcv (A1), mtn (A2), trihead (A3).
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

MAX_LAG = 15


# ---------------------------------------------------------------- tiện ích chung
def topk_pool(scores, mask, fraction=0.1):
    """Trung bình top-k (10%) điểm hợp lệ mỗi clip; [B, T] → [B]."""
    out = []
    for s, m in zip(scores, mask):
        v = s[m]
        if not len(v):
            out.append(s.new_tensor(0.0))
            continue
        out.append(v.topk(max(1, math.ceil(len(v) * fraction))).values.mean())
    return torch.stack(out)


def pool_scores(scores, mask, how):
    if how == "lse":
        return torch.logsumexp(scores.masked_fill(~mask, -1e4), dim=1)
    if how == "mean":
        return (scores * mask).sum(1) / mask.sum(1).clamp(min=1)
    return topk_pool(scores.sigmoid(), mask)


def masked_bce(logits, target, mask, pos_weight=None):
    if not mask.any():
        return logits.sum() * 0
    weight = None
    if pos_weight is not None:
        weight = torch.where(target > 0.5, pos_weight, torch.ones_like(target))[mask]
    return F.binary_cross_entropy_with_logits(logits[mask], target[mask], weight=weight)


class TemporalBlock(nn.Module):
    """Conv1d giãn có residual trên [B, T, C] (mask: frame ngoài clip bị đặt 0)."""

    def __init__(self, channels, dilations=(1, 2, 4), dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(channels, channels, 3, padding=d, dilation=d),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for d in dilations
        )

    def forward(self, x, mask):
        z = (x * mask[..., None]).transpose(1, 2)
        for block in self.blocks:
            z = z + block(z)
        return z.transpose(1, 2) * mask[..., None]


def shift(x, delta):
    """y[:, t] = x[:, t + delta] (đệm 0 ngoài biên); x [B, T, ...]."""
    if delta == 0:
        return x
    out = torch.zeros_like(x)
    if delta > 0:
        out[:, :-delta] = x[:, delta:]
    else:
        out[:, -delta:] = x[:, :delta]
    return out


class Model(nn.Module):
    needs = {"av"}
    heads = {"score": "ai"}
    pool = "topk"
    trainable = True
    references = 0
    crop_frames = None

    def loss(self, batch, out, pos_weight=None):
        total = 0
        mask = batch["mask"]
        self.loss_parts = {}  # từng thành phần (frame / video theo tác vụ) để ghi log
        for head, task in self.heads.items():
            target = batch[f"y_{task}"]
            pw = None if pos_weight is None else pos_weight.get(task)
            frame = masked_bce(out[head], target, mask, pw)
            video = ((target * mask).amax(1) > 0).float()
            score = topk_pool(out[head].sigmoid(), mask).clamp(1e-6, 1 - 1e-6)
            clip = F.binary_cross_entropy(score, video)
            self.loss_parts[f"{task}_frame"] = float(frame.detach())
            self.loss_parts[f"{task}_video"] = float(clip.detach())
            total = total + frame + clip
        return total


# ---------------------------------------------------------------- baseline
class SpeechForensics(Model):
    heads = {"score": "ai"}
    pool = "mean"
    trainable = False

    def __init__(self, **_):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, batch):
        a, v, mask = F.normalize(batch["a"], dim=-1), F.normalize(batch["v"], dim=-1), batch["mask"]
        best, best_sim = None, None
        for delta in range(-MAX_LAG, MAX_LAG + 1):
            sim = (a * shift(v, delta)).sum(-1) * mask * shift(mask.float(), delta)
            mean = sim.sum(1) / mask.sum(1).clamp(min=1)
            if best is None:
                best, best_sim = mean, sim
            else:
                better = mean > best
                best = torch.where(better, mean, best)
                best_sim = torch.where(better[:, None], sim, best_sim)
        return {"score": -best_sim}


class AVHAlign(Model):
    pool = "lse"

    def __init__(self, dim, supervised=False, **_):
        super().__init__()
        self.supervised = supervised
        self.heads = {"score": "ai"}
        layers, width = [], 2 * dim
        for out in (512, 256, 128):
            layers += [nn.Linear(width, out), nn.LayerNorm(out), nn.ReLU()]
            width = out
        self.phi = nn.Sequential(*layers, nn.Linear(width, 1))

    def pair(self, a, v):
        return self.phi(torch.cat([a, v], -1))[..., 0]

    def forward(self, batch):
        a, v = F.normalize(batch["a"], dim=-1), F.normalize(batch["v"], dim=-1)
        return {"score": -self.pair(a, v), "_a": a, "_v": v}

    def loss(self, batch, out, pos_weight=None):
        mask = batch["mask"]
        if self.supervised:
            score = torch.logsumexp(out["score"].masked_fill(~mask, -1e4), dim=1)
            return F.binary_cross_entropy_with_logits(score, batch["video"]["ai"].to(score))
        real = batch["video"]["ai"] == 0
        if not real.any():
            return out["score"].sum() * 0
        a, v, m = out["_a"][real], out["_v"][real], mask[real]
        logits = torch.stack(
            [
                self.pair(a, shift(v, d)).masked_fill(~shift(m, d), -1e4)
                for d in range(-MAX_LAG, MAX_LAG + 1)
            ],
            dim=-1,
        )
        loss = -(logits.log_softmax(-1)[..., MAX_LAG])
        return (loss * m).sum() / m.sum().clamp(min=1)


def lip_kinematics(lips):
    """[B, T, 20, 2] → y và 3 bậc sai phân [B, T, 80]."""
    y = lips[..., 1]
    orders = [y]
    for _ in range(3):
        prev = orders[-1]
        orders.append(torch.cat([torch.zeros_like(prev[:, :1]), prev[:, 1:] - prev[:, :-1]], 1))
    return torch.cat(orders, -1)


REGIONS = ((0, 7), (6, 12), (12, 17), (16, 20))  # môi ngoài trên / dưới, môi trong trên / dưới


class BioLip(Model):
    needs = {"mouth"}
    heads = {"score": "visual"}
    window, stride = 25, 5

    def __init__(self, **_):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(80, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.temporal_proj = nn.Linear(192, 128)
        self.region = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 32))
        self.stat = nn.Sequential(nn.Linear(80, 128), nn.GELU(), nn.Linear(128, 64))
        self.head = nn.Sequential(
            nn.Linear(224, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, batch):
        x = lip_kinematics(batch["lips"])
        B, T, _ = x.shape
        if T < self.window:
            x = F.pad(x, (0, 0, 0, self.window - T))
        windows = x.unfold(1, self.window, self.stride).transpose(2, 3)  # [B, W, 25, 80]
        W = windows.shape[1]
        flat = windows.reshape(B * W, self.window, 80)
        h = self.temporal(flat.transpose(1, 2))
        pooled = torch.cat([h.mean(-1), h.amax(-1), h.std(-1)], -1)
        temporal = self.temporal_proj(pooled)
        y = flat[..., :20]
        velocity = flat[..., 20:40]
        amp = torch.stack([y[..., a:b].std(1).mean(-1) for a, b in REGIONS], -1)
        vel = torch.stack([velocity[..., a:b].abs().mean((1, 2)) for a, b in REGIONS], -1)
        region = self.region(
            torch.cat(
                [
                    amp / amp.sum(-1, keepdim=True).clamp(min=1e-6),
                    vel / vel.sum(-1, keepdim=True).clamp(min=1e-6),
                ],
                -1,
            )
        )
        stat = self.stat(flat.std(1))
        logit = self.head(torch.cat([temporal, region, stat], -1)).view(B, W)
        # Mỗi frame nhận trung bình logit các cửa sổ phủ nó.
        frames = torch.zeros(B, max(T, self.window), device=x.device)
        count = torch.zeros_like(frames)
        for w in range(W):
            s = w * self.stride
            frames[:, s : s + self.window] += logit[:, w : w + 1]
            count[:, s : s + self.window] += 1
        return {"score": (frames / count.clamp(min=1))[:, :T]}


class AVLipSyncPlus(Model):
    def __init__(self, dim, hidden=128, dropout=0.1, **_):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(2 * dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.temporal = TemporalBlock(hidden, dropout=dropout)
        self.out = nn.Linear(hidden, 1)

    def forward(self, batch):
        h = self.temporal(self.proj(torch.cat([batch["a"], batch["v"]], -1)), batch["mask"].float())
        return {"score": self.out(h)[..., 0]}


class AudioTrunk(nn.Module):
    """XLS-R (đóng băng, đã cache) → AASIST-lite: chiếu + 2 lớp self-attention theo thời gian."""

    def __init__(self, hidden=128, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(1024, hidden), nn.LayerNorm(hidden), nn.GELU())
        layer = nn.TransformerEncoderLayer(hidden, 4, 2 * hidden, dropout, batch_first=True)
        self.attention = nn.TransformerEncoder(layer, 2)

    def forward(self, ssl, mask):
        return self.attention(self.proj(ssl), src_key_padding_mask=~mask)


class XLSRAasist(Model):
    needs = {"ssl"}
    heads = {"audio": "audio"}

    def __init__(self, hidden=128, dropout=0.1, **_):
        super().__init__()
        self.trunk = AudioTrunk(hidden, dropout)
        self.out = nn.Linear(hidden, 1)

    def forward(self, batch):
        return {"audio": self.out(self.trunk(batch["ssl"], batch["mask"]))[..., 0]}


# ---------------------------------------------------------------- A1 · AVCV
class SyncTrunk(nn.Module):
    """Cost volume tiếng-môi theo độ trễ ±15 → CNN 2D (thời gian × độ trễ) → đặc trưng frame.

    learned=False: cos của feature gốc (không chiếu học). conv2d=False: MLP trên hồ sơ độ trễ
    từng frame (như TriCue v002).
    """

    def __init__(self, dim, hidden=128, learned=True, conv2d=True, dropout=0.1):
        super().__init__()
        self.learned, self.conv2d = learned, conv2d
        lags = 2 * MAX_LAG + 1
        if learned:
            self.pa = nn.Sequential(nn.Linear(dim, 256), nn.LayerNorm(256))
            self.pv = nn.Sequential(nn.Linear(dim, 256), nn.LayerNorm(256))
        channels = 2 if learned else 1
        if conv2d:
            self.cost = nn.Sequential(
                nn.Conv2d(channels, 32, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 32, 3, padding=(2, 1), dilation=(2, 1)),
                nn.GELU(),
                nn.Conv2d(32, 32, 3, padding=(4, 1), dilation=(4, 1)),
                nn.GELU(),
            )
            self.lag_score = nn.Conv2d(32, 1, 1)
            feature = 64 + 3
        else:
            self.profile = nn.Sequential(
                nn.Linear(channels * lags, 128), nn.GELU(), nn.Linear(128, 64)
            )
            feature = 64
        self.content = nn.Sequential(nn.Linear(2 * dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.mix = nn.Sequential(
            nn.Linear(feature + hidden, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.temporal = TemporalBlock(hidden, dropout=dropout)
        self.register_buffer("lags", torch.arange(-MAX_LAG, MAX_LAG + 1).float())

    def volume(self, a, v, mask):
        cos_a, cos_v = F.normalize(a, dim=-1), F.normalize(v, dim=-1)
        layers = [
            torch.stack(
                [(cos_a * shift(cos_v, d)).sum(-1) for d in range(-MAX_LAG, MAX_LAG + 1)], -1
            )
        ]
        if self.learned:
            pa, pv = self.pa(a), self.pv(v)
            layers.insert(
                0,
                torch.stack(
                    [(pa * shift(pv, d)).sum(-1) / 16.0 for d in range(-MAX_LAG, MAX_LAG + 1)], -1
                ),
            )
        valid = (
            torch.stack([shift(mask, d) for d in range(-MAX_LAG, MAX_LAG + 1)], -1)
            & mask[..., None]
        )
        return torch.stack(layers, 1) * valid[:, None], valid  # [B, C, T, L]

    def forward(self, a, v, mask):
        volume, valid = self.volume(a, v, mask)
        if self.conv2d:
            h = self.cost(volume)  # [B, 32, T, L]
            score = self.lag_score(h)[:, 0].masked_fill(~valid, -1e4)  # [B, T, L]
            weights = score.softmax(-1)
            lag = (weights * self.lags).sum(-1, keepdim=True) / MAX_LAG
            peak = score.amax(-1, keepdim=True).clamp(-20, 20) / 10
            entropy = -(weights * weights.clamp(min=1e-8).log()).sum(-1, keepdim=True) / math.log(
                len(self.lags)
            )
            pooled = torch.cat([h.mean(-1), h.amax(-1)], 1).transpose(1, 2)  # [B, T, 64]
            feature = torch.cat([pooled, lag, peak, entropy], -1)
        else:
            B, C, T, L = volume.shape
            feature = self.profile(volume.permute(0, 2, 1, 3).reshape(B, T, C * L))
        h = self.mix(torch.cat([feature, self.content(torch.cat([a, v], -1))], -1))
        return self.temporal(h, mask.float())


class AVCV(Model):
    heads = {"mismatch": "mismatch", "visual": "visual"}

    def __init__(self, dim, hidden=128, dropout=0.1, learned=True, conv2d=True, **_):
        super().__init__()
        self.trunk = SyncTrunk(dim, hidden, learned, conv2d, dropout)
        self.out = nn.Linear(hidden, 2)

    def forward(self, batch):
        logits = self.out(self.trunk(batch["a"], batch["v"], batch["mask"]))
        return {"mismatch": logits[..., 0], "visual": logits[..., 1]}


# ---------------------------------------------------------------- A2 · MTN
def trace_streams(mouth, npr=True, temporal=True):
    """[B, T, H, W] uint8 → [B, C, T, H, W]: ảnh, residual NPR, residual thời gian."""
    x = mouth.float() / 255.0
    streams = [x - 0.5]
    if npr:
        down = F.avg_pool2d(x.flatten(0, 1)[:, None], 2)
        up = F.interpolate(down, scale_factor=2, mode="nearest")[:, 0].view_as(x)
        streams.append((x - up) * 4)
    if temporal:
        prev = torch.cat([x[:, :1], x[:, :-1]], 1)
        nxt = torch.cat([x[:, 1:], x[:, -1:]], 1)
        streams.append((x - (prev + nxt) / 2) * 4)
    return torch.stack(streams, 1)


class TraceTrunk(nn.Module):
    def __init__(self, hidden=128, npr=True, temporal=True, kinematics=True, dropout=0.1):
        super().__init__()
        self.npr, self.temporal_res, self.kinematics = npr, temporal, kinematics
        channels = 1 + int(npr) + int(temporal)
        self.cnn = nn.Sequential(
            nn.Conv3d(channels, 24, 3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(24),
            nn.GELU(),
            nn.Conv3d(24, 48, 3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(48),
            nn.GELU(),
            nn.Conv3d(48, 64, 3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),
        )
        width = 128 + (64 if kinematics else 0)
        if kinematics:
            self.kin = nn.Sequential(nn.Linear(60, 64), nn.GELU())
        self.mix = nn.Sequential(nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.temporal = TemporalBlock(hidden, dropout=dropout)

    def forward(self, mouth, lips, mask):
        h = self.cnn(trace_streams(mouth, self.npr, self.temporal_res))  # [B, 64, T, h, w]
        feature = torch.cat([h.mean((-1, -2)), h.amax((-1, -2))], 1).transpose(1, 2)
        if self.kinematics:
            feature = torch.cat([feature, self.kin(lip_kinematics(lips)[..., :60])], -1)
        return self.temporal(self.mix(feature), mask.float())


class MTN(Model):
    needs = {"mouth"}
    heads = {"visual": "visual"}
    crop_frames = 75

    def __init__(self, hidden=128, dropout=0.1, npr=True, temporal=True, kinematics=True, **_):
        super().__init__()
        self.trunk = TraceTrunk(hidden, npr, temporal, kinematics, dropout)
        self.out = nn.Linear(hidden, 1)

    def forward(self, batch):
        return {
            "visual": self.out(self.trunk(batch["mouth"], batch["lips"], batch["mask"]))[..., 0]
        }


# ---------------------------------------------------------------- A3 · TriHead-Ref
class FiLM(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.gamma_beta = nn.Linear(hidden, 2 * hidden)
        nn.init.zeros_(self.gamma_beta.weight)
        nn.init.zeros_(self.gamma_beta.bias)

    def forward(self, h, ref):
        gamma, beta = self.gamma_beta(ref).chunk(2, -1)
        return torch.cat([h * (1 + gamma[:, None]) + beta[:, None], h - ref[:, None]], -1)


class TriHead(Model):
    """Ba nhánh (sync A1, trace A2, audio XLS-R) → 3 đầu; tham chiếu người qua FiLM."""

    crop_frames = 75

    def __init__(
        self,
        dim,
        hidden=128,
        dropout=0.1,
        use_ref=True,
        use_audio=True,
        use_trace=True,
        use_sync=True,
        **_,
    ):
        super().__init__()
        self.use = {"ref": use_ref, "audio": use_audio, "trace": use_trace, "sync": use_sync}
        self.needs = (
            ({"av"} if use_sync else set())
            | ({"mouth"} if use_trace else set())
            | ({"ssl"} if use_audio else set())
        )
        self.references = 3 if use_ref else 0
        self.heads = {}
        if use_sync:
            self.sync = SyncTrunk(dim, hidden, dropout=dropout)
        if use_trace:
            self.trace = TraceTrunk(hidden, dropout=dropout)
        if use_audio:
            self.audio = AudioTrunk(hidden, dropout)
        width = 2 * hidden if use_ref else hidden
        if use_ref:
            self.film = nn.ModuleDict(
                {k: FiLM(hidden) for k in ("sync", "trace", "audio") if self.use[k]}
            )
        visual_in = width * (int(use_trace) + int(use_sync))
        if visual_in:
            self.visual_head = nn.Sequential(
                nn.Linear(visual_in, hidden), nn.GELU(), nn.Linear(hidden, 1)
            )
            self.heads["visual"] = "visual"
        if use_audio:
            self.audio_head = nn.Sequential(
                nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 1)
            )
            self.heads["audio"] = "audio"
        mismatch_in = width * (int(use_sync) + int(use_audio))
        if mismatch_in:
            self.mismatch_head = nn.Sequential(
                nn.Linear(mismatch_in, hidden), nn.GELU(), nn.Linear(hidden, 1)
            )
            self.heads["mismatch"] = "mismatch"

    def branches(self, batch):
        mask = batch["mask"]
        out = {}
        if self.use["sync"]:
            out["sync"] = self.sync(batch["a"], batch["v"], mask)
        if self.use["trace"]:
            out["trace"] = self.trace(batch["mouth"], batch["lips"], mask)
        if self.use["audio"]:
            out["audio"] = self.audio(batch["ssl"], mask)
        return out

    def reference(self, batch):
        """Trung bình (theo thời gian rồi theo các clip tham chiếu) đặc trưng nhánh; không grad."""
        refs = batch["refs"]
        B, R, T = refs["valid"].shape
        flat = {k: v.flatten(0, 1) for k, v in refs.items()}
        flat["mask"] = flat.pop("valid")
        with torch.no_grad():
            feats = self.branches(flat)
        count = batch["ref_count"].clamp(min=1)
        out = {}
        for name, h in feats.items():
            m = flat["mask"].float()[..., None]
            per_clip = ((h * m).sum(1) / m.sum(1).clamp(min=1)).view(B, R, -1)
            present = (torch.arange(R, device=h.device)[None] < batch["ref_count"][:, None]).float()
            out[name] = (per_clip * present[..., None]).sum(1) / count[:, None]
        return out

    def forward(self, batch):
        feats = self.branches(batch)
        if self.use["ref"]:
            if "refs" in batch and batch["ref_count"].max() > 0:
                refs = self.reference(batch)
            else:
                refs = {k: torch.zeros_like(h[:, 0]) for k, h in feats.items()}
            feats = {k: self.film[k](h, refs[k]) for k, h in feats.items()}
        out = {}
        if "visual" in self.heads:
            out["visual"] = self.visual_head(
                torch.cat([feats[k] for k in ("trace", "sync") if k in feats], -1)
            )[..., 0]
        if "audio" in self.heads:
            out["audio"] = self.audio_head(feats["audio"])[..., 0]
        if "mismatch" in self.heads:
            out["mismatch"] = self.mismatch_head(
                torch.cat([feats[k] for k in ("sync", "audio") if k in feats], -1)
            )[..., 0]
        return out


# ---------------------------------------------------------------- registry
SPECS = {
    "speechforensics": (SpeechForensics, {}),
    "avh_align": (AVHAlign, {}),
    "avh_align_sup": (AVHAlign, {"supervised": True}),
    "biolip": (BioLip, {}),
    "av_lipsync_plus": (AVLipSyncPlus, {}),
    "xlsr_aasist": (XLSRAasist, {}),
    "avcv": (AVCV, {}),
    "avcv_no2d": (AVCV, {"conv2d": False}),
    "avcv_cos": (AVCV, {"learned": False}),
    "mtn": (MTN, {}),
    "mtn_no_npr": (MTN, {"npr": False}),
    "mtn_no_temporal": (MTN, {"temporal": False}),
    "mtn_no_kin": (MTN, {"kinematics": False}),
    "mtn_no_aug": (MTN, {}),
    "trihead": (TriHead, {}),
    "trihead_no_ref": (TriHead, {"use_ref": False}),
    "trihead_no_audio": (TriHead, {"use_audio": False}),
    "trihead_no_trace": (TriHead, {"use_trace": False}),
    "trihead_no_sync": (TriHead, {"use_sync": False}),
}
NO_AUGMENT = {"mtn_no_aug"}
# Dịch tiếng so với hình khi train (data.lag_audio) cho mọi model có cả hai phía; trừ avh_align
# không giám sát: nó học "độ lệch 0 = cặp đúng" từ real, dịch tiếng làm sai mục tiêu InfoNCE.
NO_LAG_AUGMENT = {"avh_align"}


def build(name, dim=1024, hidden=128, dropout=0.1):
    cls, options = SPECS[name]
    model = cls(dim=dim, hidden=hidden, dropout=dropout, **options)
    model.name = name
    model.augment = "mouth" in model.needs and name not in NO_AUGMENT
    both_sides = "av" in model.needs or {"ssl", "mouth"} <= set(model.needs)
    model.lag_augment = both_sides and model.trainable and name not in NO_LAG_AUGMENT
    return model
