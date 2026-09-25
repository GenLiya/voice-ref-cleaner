#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voice-ref-cleaner —— 从访谈 / 视频音频里，做出"干净的单人语音参考素材"（供音色克隆 / TTS 用）。

它做三件事，第 1 件是别人的，2、3 件是本项目加的：

  1) 去 BGM：调用 **Demucs**（facebookresearch/demucs，Alexandre Défossez 等，MIT）做音源分离。
     ⚠️ 版权与出处：https://github.com/facebookresearch/demucs
        论文：Défossez et al., "Hybrid Transformers for Music Source Separation", ICASSP 2023.
        模型权重：htdemucs（首次运行会自动下载）。**本项目不修改 Demucs，只调用它。**

  2) 按说话人分离：用 **CAM++（3D-Speaker / ModelScope，iic/speech_campplus_sv_zh-cn_16k-common）**
     算 192 维声纹向量后聚类，把"主持人 / 画外音"和目标说话人分开。
     ⚠️ 出处：https://github.com/modelscope/3D-Speaker（Apache-2.0）

  3) 去音效 / 非语音（本项目新增）：VAD 按句切分 + 多维特征筛查，
     把"叮、咚、咻、嘿"这类音效瞬态、以及别人的声音挡掉。

为什么需要 3)：只做 Demucs 是不够的 —— 实测人声轨里仍会残留大量音效瞬态，
而音色克隆的参考素材里一旦混进这些，克隆出来的音色就会被污染。

用法见 README.md。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
EPS = 1e-9

# ─────────────────────────── 基础工具 ───────────────────────────


def run(cmd: list[str], quiet: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-1500:]
        raise RuntimeError(f"命令失败（{cmd[0]}）：\n{tail}")
    if not quiet:
        print("  $ " + " ".join(cmd))
    return r


def load_wav(path: Path | str, sr: int = SR, mono: bool = True) -> tuple[np.ndarray, int]:
    w, s = sf.read(str(path), always_2d=False)
    if w.ndim > 1 and mono:
        w = w.mean(axis=1)
    w = w.astype(np.float32)
    if s != sr:
        import librosa

        w = librosa.resample(w, orig_sr=s, target_sr=sr)
        s = sr
    return w, s


def to_wav(src: Path, dst: Path, sr: int = 44100, mono: bool = False) -> Path:
    """用 ffmpeg 从任意音视频里抽音频。"""
    ch = "1" if mono else "2"
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", ch, "-ar", str(sr), str(dst)])
    return dst


# ─────────────────────────── ① 去 BGM（Demucs） ───────────────────────────


def demucs_vocals(src: Path, outdir: Path, model: str = "htdemucs", device: str = "cpu") -> tuple[Path, Path]:
    """用 Demucs 分离人声与伴奏。

    返回 (vocals, no_vocals)。**这两个文件都留着** —— 第 3 步会用两者的能量比
    判断某个片段到底是不是"人在说话"（音效/音乐的伴奏能量往往高于人声）。
    """
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / model / src.stem
    voc, nov = stem / "vocals.wav", stem / "no_vocals.wav"
    # 复用上次的分离结果：Demucs 在 CPU 上约等于实时的 1.5 倍，每改一次参数重跑一遍很浪费时间
    if voc.exists() and nov.exists():
        print("      复用已有人声分离结果（要重跑就删掉 " + str(stem) + "）")
        return voc, nov
    cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", model, "-d", device, "-o", str(outdir), str(src)]
    try:
        run(cmd)
    except RuntimeError as e:
        if "No module named" in str(e):
            raise SystemExit("✗ 需要先装 Demucs：pip install demucs\n  （或加 --no-demucs 跳过这一步）") from e
        raise
    if not voc.exists():
        raise SystemExit(f"✗ Demucs 没产出 {voc}")
    return voc, nov


# ─────────────────────────── ② 声纹向量与分簇（CAM++） ───────────────────────────

CAMPP_MODEL = "iic/speech_campplus_sv_zh-cn_16k-common"


def _campp():
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
    except ImportError as e:
        raise SystemExit(
            "✗ 需要 ModelScope 的 CAM++ 说话人确认模型：\n"
            "    pip install modelscope addict 'datasets<4'\n"
            "  （或加 --no-speaker 跳过说话人分离）"
        ) from e
    return pipeline(task=Tasks.speaker_verification, model=CAMPP_MODEL).model


def embed_windows(wav: np.ndarray, sr: int, win: float, hop: float, model) -> tuple[np.ndarray, np.ndarray]:
    import torch

    W, H = int(win * sr), int(hop * sr)
    starts = list(range(0, max(1, len(wav) - W + 1), H))
    out = np.zeros((len(starts), 192), dtype=np.float32)
    with torch.no_grad():
        for i, s in enumerate(starts):
            e = model(torch.from_numpy(wav[s : s + W].copy()).unsqueeze(0)).squeeze(0).numpy().astype(np.float32)
            out[i] = e / (np.linalg.norm(e) + EPS)
    return out, np.array(starts, dtype=int)


def embed_one(seg: np.ndarray, model) -> np.ndarray:
    import torch

    with torch.no_grad():
        e = model(torch.from_numpy(seg.copy()).unsqueeze(0)).squeeze(0).numpy().astype(np.float32)
    return e / (np.linalg.norm(e) + EPS)


def cluster_speakers(emb: np.ndarray, voiced: np.ndarray, k: int = 0) -> tuple[np.ndarray, int, float]:
    """对有声窗的声纹向量聚类。k=0 时在 2..6 里按轮廓系数自动选。"""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    v = emb[voiced]
    best = None
    for kk in ([k] if k else range(2, 7)):
        if len(v) < kk + 2:
            continue
        lab = KMeans(n_clusters=kk, n_init=10, random_state=0).fit_predict(v)
        if len(set(lab)) < 2:
            continue
        sc = silhouette_score(v, lab)
        if best is None or sc > best[2]:
            best = (lab, kk, sc)
    if best is None:
        raise SystemExit("✗ 聚类失败（音频太短或没有有声段）")
    lab, kk, sc = best
    # 簇编号按"首次出现时间"固定 —— 否则每次跑同一个说话人可能换字母，用户按字母反馈会错位
    first: dict[int, int] = {}
    for i, wi in enumerate(np.where(voiced)[0]):
        first.setdefault(int(lab[i]), int(wi))
    remap = {c: r for r, (c, _) in enumerate(sorted(first.items(), key=lambda kv: kv[1]))}
    return np.array([remap[int(x)] for x in lab]), kk, sc


# ─────────────────────────── ③ 去音效（本项目新增） ───────────────────────────


@dataclass
class Chunk:
    start: float
    dur: float
    feats: dict = field(default_factory=dict)
    score: float = 0.0
    kept: bool = False
    reason: str = ""


def _voiced_ratio(seg: np.ndarray, sr: int, lo: float = 80.0, hi: float = 260.0) -> float:
    """稳定基频帧占比。语音有基频；"咻/咚"这类音效没有。"""
    import librosa

    if len(seg) < 2048:
        return 0.0
    f0 = librosa.yin(seg, fmin=70, fmax=350, sr=sr, frame_length=1024, hop_length=160)
    return float(((f0 > lo) & (f0 < hi)).mean())


def _flatness(seg: np.ndarray, sr: int) -> float:
    """频谱平坦度（谱几何均值/算术均值）。语音谐波结构 → 低；宽带噪声/音效 → 高。"""
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / sr)
    S = S[(f > 300) & (f < 8000)] + EPS
    return float(np.exp(np.mean(np.log(S))) / np.mean(S))


def _crest_ratio(seg: np.ndarray, sr: int) -> float:
    """谱峰度：最大谱峰 / 中位谱。**"叮、噔"这类音效是窄带强音，这个值会很高。**"""
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / sr)
    S = S[(f > 300) & (f < 8000)] + EPS
    return float(np.percentile(S, 99.9) / (np.median(S) + EPS))


def _peak_ratio(seg: np.ndarray) -> float:
    """峰均比：短时能量的峰值/中位。爆音（"嘿"）很尖 → 高。"""
    rms = np.sqrt(np.convolve(seg**2, np.ones(400) / 400, mode="same"))
    return float(rms.max() / (np.median(rms) + EPS))


def _vocal_dominance(voc: np.ndarray, nov: np.ndarray, sr: int) -> float:
    """人声轨能量 / 伴奏轨能量（dB 差）。音效、音乐残留的片段这个值会偏低。"""
    n = min(len(voc), len(nov))
    if n == 0:
        return 0.0
    ev = float(np.sqrt((voc[:n] ** 2).mean()) + EPS)
    en = float(np.sqrt((nov[:n] ** 2).mean()) + EPS)
    return float(20 * np.log10(ev / en))


def vad_speech(wav: np.ndarray, sr: int, min_speech: float = 0.35, min_sil: float = 0.20) -> list[tuple[float, float]]:
    """Silero VAD 找语音区间（秒）。**先切句再挑** —— 在 1 秒网格上切，音效会被整块带进来。"""
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()
    ts = get_speech_timestamps(
        torch.from_numpy(wav), model, sampling_rate=sr,
        min_speech_duration_ms=int(min_speech * 1000),
        min_silence_duration_ms=int(min_sil * 1000),
        return_seconds=True,
    )
    return [(float(t["start"]), float(t["end"])) for t in ts]


def screen_chunks(
    voc: np.ndarray, nov: np.ndarray, sr: int, regions: list[tuple[float, float]],
    target_centroid: np.ndarray | None, model, chunk: float = 1.0, sub: float = 0.5,
) -> list[Chunk]:
    """把每个语音区间切成 <=chunk 秒的小块，逐块打分。

    ⚠️⚠️ 两条是被实测教训逼出来的，别改回去：

    1) **块要短（默认 1 秒），而且特征要取"子窗里最差的那个"，不能取平均。**
       第一版用 3 秒块 + 平均特征 → 一个 0.3 秒的音效只占这块的 10%，平均分照样很高 →
       整块被放行，成品里全是"噔噔""鸭叫"。实测用户原话："这次做出来的还没上次好，
       上次至少没有别人的声音"。改成 1 秒块 + **最差子窗**之后，音效藏不住了。

    2) **声纹相似度要在"比块更长的窗"上算**（块 ±0.25s）—— 1 秒太短，
       说话人向量不稳；但**音效类特征必须留在这 1 秒里算**，否则又变成平均。

    打分＝各特征**排名求和**：
      相似度 ↑、人声占比 ↑、基频稳定度 ↑、平坦度 ↓、谱峰度 ↓、峰均比 ↓
    """
    out: list[Chunk] = []
    for (a, b) in regions:
        t = a
        while t + 0.3 < b:
            d = min(chunk, b - t)
            seg = voc[int(t * sr) : int((t + d) * sr)]
            if len(seg) < int(0.3 * sr):
                break
            f = _worst_subwindow(voc, nov, sr, t, d, model, target_centroid, sub)
            if f is not None:
                out.append(Chunk(start=t, dur=d, feats=f))
            t += d
    if not out:
        return out
    keys = ["sim", "vdom", "voiced", "flat", "crest", "peak"]
    ranks = {}
    for k in keys:
        vals = [(-c.feats[k] if _HIGHER[k] else c.feats[k]) for c in out]
        ranks[k] = np.argsort(np.argsort(vals))
    for i, c in enumerate(out):
        c.score = float(sum(ranks[k][i] for k in keys))
    smax = max(c.score for c in out)
    for c in out:
        c.score = 1.0 - c.score / (smax + EPS)
    return out


# 「越大越好」的特征；其余越小越好。_worst_subwindow 用它决定取 min 还是 max。
_HIGHER = {"sim": True, "vdom": True, "voiced": True, "flat": False, "crest": False, "peak": False}


def _unit_feats(voc, nov, sr, t, dur, model, centroid):
    seg = voc[int(t * sr) : int((t + dur) * sr)]
    if len(seg) < int(0.15 * sr):
        return None
    f = dict(
        flat=_flatness(seg, sr),
        crest=_crest_ratio(seg, sr),
        peak=_peak_ratio(seg),
        voiced=_voiced_ratio(seg, sr),
        vdom=_vocal_dominance(voc[int(t * sr) : int((t + dur) * sr)],
                              nov[int(t * sr) : int((t + dur) * sr)], sr),
    )
    if model is not None and centroid is not None:
        c0, c1 = max(0, int((t - 0.25) * sr)), min(len(voc), int((t + dur + 0.25) * sr))
        f["sim"] = float(np.dot(embed_one(voc[c0:c1], model), centroid))
    else:
        f["sim"] = 0.0
    return f


def _worst_subwindow(voc, nov, sr, t, dur, model, centroid, sub=0.5, floor_db: float = -45.0):
    """把一块切成 sub 秒的子窗（步长 sub/2），逐子窗算特征，取**最差**的那个。

    这一步是"音效藏在块里"的唯一解药：段里只要有一处像音效，整块就被判差、丢掉。

    ⚠️ **静音子窗必须跳过**：拼接出来的参考素材在块间有静音间隔，
    数字静音的平坦度恰好是 1.0、基频 0 —— 不排除的话，自检会把"块间空隙"全报成疑似音效
    （实测踩过：自检报的 10 段全是空隙，掩盖了真正的问题）。
    """
    out, tt, step = None, t, sub / 2
    while tt < t + dur - 1e-6:
        d = min(sub, t + dur - tt)
        if d < 0.15:
            break
        seg = voc[int(tt * sr) : int((tt + d) * sr)]
        rms_db = 20 * np.log10(max(float(np.sqrt((seg ** 2).mean())), 1e-9))
        if rms_db > floor_db:                      # 只统计"有声音"的子窗
            f = _unit_feats(voc, nov, sr, tt, d, model, centroid)
            if f:
                if out is None:
                    out = dict(f)
                else:
                    for k, v in f.items():
                        out[k] = min(out[k], v) if _HIGHER[k] else max(out[k], v)
        tt += step
    return out


HARD_RULES = [
    ("sim", "<", 0.45, "声纹不像目标说话人"),
    ("crest", ">", 3e5, "谱峰度过高（窄带强音＝音效）"),
    ("peak", ">", 15.0, "峰均比过高（爆音）"),
    ("flat", ">", 0.40, "平坦度过高（噪声）"),
]


def hard_flags(units: list[dict], has_target: bool) -> list[dict]:
    """**绝对阈值**判定，不是百分位 —— 这是它能当循环终止条件的原因。

    用百分位的话"最差的 15%"永远存在，循环永远不收敛（实测踩过）。
    """
    out = []
    for c in units:
        why = [msg for k, op, thr, msg in HARD_RULES
               if not (k == "sim" and not has_target)
               and ((c[k] < thr) if op == "<" else (c[k] > thr))]
        if why:
            out.append(dict(c, why=why))
    return sorted(out, key=lambda c: c["t"])


def qc_reference(path: Path, sr: int, unit: float, model, centroid,
                 period: float | None = None) -> list[dict]:
    """对**成品参考素材**再做一遍同样的筛查，返回**每一格**的特征。

    ⚠️ 两个坑（都实测踩过）：
      1) **网格必须和拼接时的网格对齐**（块长 + 块间静音）。否则自检的每一格都跨在两个块上，
         取"最差子窗"就等于把两个块的缺点算到一格头上，会报出一堆假警报。
      2) **`vdom`（人声÷伴奏能量比）在成品上没意义** —— 成品里没有伴奏轨，
         分母趋零 → 这个特征会变成同一个巨大数值。自检**不能用它**排名。
    """
    per = period or (unit + 0.25)
    y, _ = load_wav(path)
    keys = ["sim", "voiced", "flat", "crest", "peak"]      # 不含 vdom
    units = []
    i = 0
    while (i * per + unit) <= len(y) / sr + 1e-6:
        t = i * per
        seg = y[int(t * sr) : int((t + unit) * sr)]
        if 20 * np.log10(max(float(np.sqrt((seg ** 2).mean())), 1e-9)) < -50:
            i += 1
            continue
        f = _worst_subwindow(y, np.zeros_like(y), sr, t, unit, model, centroid, unit / 2)
        if f:
            units.append(dict(t=t, **{k: f[k] for k in keys}))
        i += 1
    return units


# ─────────────────────────── 主流程 ───────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="从访谈/视频音频做出干净的单人语音参考素材")
    ap.add_argument("input", help="输入音频或视频文件")
    ap.add_argument("-o", "--out", required=True, help="输出目录")
    ap.add_argument("--target", default="auto",
                    help="目标说话人：auto=自动挑；A/B/C…=指定分簇字母；或给一个参考音频路径（按声纹匹配）")
    ap.add_argument("--sample", type=float, default=40.0, help="参考素材目标时长（秒）")
    ap.add_argument("--keep-pct", type=float, default=55.0, help="按打分保留前百分之多少的块（默认 55）")
    ap.add_argument("--chunk", type=float, default=1.0,
                    help="切块长度（秒），默认 1.0。⚠️ 别调大：块越长，藏在块里的音效越容易被平均掉")
    ap.add_argument("--sub", type=float, default=0.5,
                    help="块内子窗长度（秒），默认 0.5；特征取子窗里**最差**的那个")
    ap.add_argument("--no-demucs", action="store_true", help="跳过去 BGM（输入已经是干净人声时用）")
    ap.add_argument("--no-speaker", action="store_true", help="跳过说话人分离（只做去 BGM + 去音效）")
    ap.add_argument("--device", default="cpu", help="demucs 设备")
    ap.add_argument("--win", type=float, default=3.0, help="声纹滑窗（秒）")
    ap.add_argument("--hop", type=float, default=1.0, help="声纹滑窗步长（秒）")
    ap.add_argument("--exclude", default="", help="要跳过的**源音频秒数**，逗号分隔（人工点名）")
    a = ap.parse_args()

    inp = Path(a.input).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not inp.exists():
        raise SystemExit(f"✗ 找不到输入：{inp}")
    exclude = {int(x) for x in a.exclude.split(",") if x.strip().isdigit()}

    # 0) 抽音频
    src44 = out / "source_44k.wav"
    print("[0/4] 抽音频 →", src44.name)
    to_wav(inp, src44, sr=44100)

    # 1) 去 BGM
    if a.no_demucs:
        voc_p, nov_p = src44, None
        print("[1/4] 跳过 Demucs（--no-demucs）")
    else:
        print("[1/4] Demucs 分离人声（首次会下载 htdemucs 权重）…")
        voc_p, nov_p = demucs_vocals(src44, out / "demucs", device=a.device)

    voc16, sr = load_wav(voc_p)
    nov16 = load_wav(nov_p)[0] if nov_p else np.zeros_like(voc16)

    # 2) VAD 切句
    print("[2/4] Silero VAD 找语音区间…")
    regions = vad_speech(voc16, sr)
    speech_s = sum(b - a_ for a_, b in regions)
    print(f"      {len(regions)} 段语音，共 {speech_s:.1f}s / 全长 {len(voc16)/sr:.1f}s")

    # 3) 说话人分离
    model, centroid, target_label = None, None, None
    if a.no_speaker:
        print("[3/4] 跳过说话人分离（--no-speaker）")
    else:
        print("[3/4] CAM++ 声纹向量聚类…")
        model = _campp()
        emb, starts = embed_windows(voc16, sr, a.win, a.hop, model)
        # 有声判定：用伴奏轨做能量门限（比固定 dB 稳）
        rms = np.array([np.sqrt((voc16[s : s + int(a.win * sr)] ** 2).mean() + EPS) for s in starts])
        voiced = rms > (np.percentile(rms, 20))
        lab, k, sc = cluster_speakers(emb, voiced, 0)
        print(f"      分簇 k={k}，轮廓系数 {sc:.3f}")
        # 每簇导出样本，供人工指认
        cents = {c: emb[voiced][lab == c].mean(axis=0) for c in np.unique(lab)}
        cents = {c: v / (np.linalg.norm(v) + EPS) for c, v in cents.items()}
        kidx = np.where(voiced)[0]
        lbl = {int(w): int(lab[i]) for i, w in enumerate(kidx)}
        for c in sorted(cents):
            sel = [int(w) for i, w in enumerate(kidx) if lab[i] == c]
            sel = sorted(sel, key=lambda w: -float(np.dot(emb[w], cents[c])))
            pure = [w for w in sel if all((w + d not in lbl) or lbl[w + d] == c for d in (-2, -1, 1, 2))]
            pieces, tot = [], 0.0
            for w in pure:
                if tot >= 30:
                    break
                s0 = int(starts[w])
                seg = voc16[s0 : s0 + int(a.hop * sr)]
                if s0 // sr in exclude:
                    continue
                pieces.append(seg); tot += a.hop
            if pieces:
                gap = np.zeros(int(0.25 * sr), dtype=np.float32)
                y = np.concatenate([np.concatenate([p, gap]) for p in pieces])
                sf.write(out / f"speaker_{chr(65+c)}.wav", y, sr)
                print(f"      speaker_{chr(65+c)}.wav  {len(y)/sr:.1f}s  ← 听这个指认目标说话人")
        # 选目标
        if a.target == "auto":
            # 没有先验时，挑"人声占比最高的那一簇"（画外音/杂音通常伴奏能量更高）
            best = None
            for c in sorted(cents):
                secs = [int(starts[int(w)]) for i, w in enumerate(kidx) if lab[i] == c]
                vd = np.mean([_vocal_dominance(voc16[s : s + int(a.win * sr)],
                                               nov16[s : s + int(a.win * sr)], sr) for s in secs[:60]])
                if best is None or vd > best[1]:
                    best = (c, vd)
            target_label, centroid = best[0], cents[best[0]]
            print(f"      自动选中 {chr(65+target_label)} 簇（人声占比 {best[1]:.1f}dB）"
                  f" —— **请听 speaker_{chr(65+target_label)}.wav 确认，不对就用 --target 指定**")
        elif a.target.upper() in list("ABCDEF") and len(a.target) == 1:
            target_label = ord(a.target.upper()) - 65
            centroid = cents[target_label]
            print(f"      目标＝{a.target.upper()} 簇")
        else:
            ref, _ = load_wav(a.target)
            centroid = embed_one(ref, model)
            print(f"      目标＝按参考音频 {Path(a.target).name} 的声纹匹配")

    # 4) 切块 + 打分 + 组装
    print("[4/4] 按句切块 → 逐块打分 → 组装参考素材")
    if model is not None and centroid is not None:
        kept_regions = []
        for r in regions:
            # 声纹过滤也别整段判：按 1 秒细查，避免"一段里夹了一句别人的话"整段被留/整段被丢
            ok = 0; tot = 0
            tt = r[0]
            while tt < r[1]:
                d = min(1.0, r[1] - tt)
                if d < 0.3:
                    break
                e = embed_one(voc16[int(tt * sr) : int((tt + d) * sr)], model)
                tot += 1
                ok += int(float(np.dot(e, centroid)) > 0.45)
                tt += d
            if tot and ok / tot > 0.5:
                kept_regions.append(r)
        print(f"      按声纹过滤后剩 {len(kept_regions)} 段 / {sum(b-a_ for a_,b in kept_regions):.1f}s")
        regions = kept_regions
    chunks = [c for c in screen_chunks(voc16, nov16, sr, regions, centroid, model, chunk=a.chunk, sub=a.sub)
              if int(c.start) not in exclude]
    if not chunks:
        raise SystemExit("✗ 没有可用片段")
    pool = sorted(chunks, key=lambda c: -c.score)
    n_keep = max(1, int(len(pool) * a.keep_pct / 100))
    for c in pool[:n_keep]:
        c.kept, c.reason = True, "保留"
    for c in pool[n_keep:]:
        c.kept, c.reason = False, "打分低"

    gap = np.zeros(int(0.25 * sr), dtype=np.float32)
    period = a.chunk + 0.25

    def assemble(picked: list[Chunk]) -> np.ndarray:
        ps = [voc16[int(c.start * sr) : int((c.start + c.dur) * sr)]
              for c in sorted(picked, key=lambda c: c.start)]
        return np.concatenate([np.concatenate([p, gap]) for p in ps]) if ps else np.zeros(sr, dtype=np.float32)

    # ── 选 → 拼 → 自检 → 剔除重拼（自纠环）────────────────────────────
    # 为什么要有这个环：**只靠"排名取前 55%"是不够的** —— 实测成品里仍混进了
    # 「声纹相似度 0.03」（根本不是目标说话人）和「谱峰度 160 万」（窄带强音＝音效）的块。
    # 排名是相对的，只能保证"比别的块好"，不能保证"本身合格"；所以要用**绝对阈值**把它们揪出来，
    # 换成池子里次优的块，循环到自检干净为止。
    banned: set[int] = set()
    ref, flagged = np.zeros(sr, dtype=np.float32), []
    for it in range(1, 6):
        picked, tot = [], 0.0
        for c in pool:
            if id(c) in banned:
                continue
            if tot >= a.sample:
                break
            picked.append(c); tot += c.dur
        ref = assemble(picked)
        tmp = out / "_qc_tmp.wav"
        sf.write(tmp, ref, sr)
        units = qc_reference(tmp, sr, a.chunk, model, centroid, period=period)
        flagged = hard_flags(units, has_target=(model is not None and centroid is not None))
        print(f"      自纠第 {it} 轮：{len(picked)} 块 / {len(ref)/sr:.1f}s → 硬性不合格 {len(flagged)} 处")
        if not flagged:
            break
        ordered = sorted(picked, key=lambda c: c.start)
        n = 0
        for f in flagged:
            i = int(round(f["t"] / period))
            if 0 <= i < len(ordered) and id(ordered[i]) not in banned:
                banned.add(id(ordered[i])); ordered[i].reason = "自检不合格：" + "/".join(f["why"]); n += 1
        if n == 0:
            break
    if (out / "_qc_tmp.wav").exists():
        (out / "_qc_tmp.wav").unlink()

    # 结算每块的状态（report.json 里要能看出"为什么留下/为什么被剔"）
    picked_ids = {id(c) for c in picked}
    for c in pool:
        c.kept = id(c) in picked_ids
        if c.kept:
            c.reason = "保留"
        elif id(c) in banned:
            pass                                   # 已在自纠环里写过具体原因
        elif c.reason == "保留":
            c.reason = "未被选上（够时长了）"
    ref_p = out / "reference_clean.wav"
    sf.write(ref_p, ref, sr)

    report = dict(
        input=str(inp), out=str(out), demucs=not a.no_demucs,
        vad_regions=len(regions), chunks_total=len(chunks), chunks_kept=len(picked),
        target=f"{chr(65+target_label)}" if target_label is not None else a.target,
        reference_seconds=round(len(ref) / sr, 2),
        selfcheck_banned=len(banned),
        chunks=[dict(start=round(c.start, 2), dur=round(c.dur, 2), score=round(c.score, 3),
                     kept=c.kept, reason=c.reason, **{k: round(v, 4) for k, v in c.feats.items()})
                for c in sorted(chunks, key=lambda c: c.start)],
    )
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✓ 参考素材 → {ref_p}  {len(ref)/sr:.1f}s（{len(picked)} 块，自纠剔除 {len(banned)} 块）")

    # ── 交付前的最后一道自检：成品按**绝对阈值**再过一遍 ──
    # 自纠环已经把不合格的块换掉了，这里再查一次，确保交付的是"自检干净"的版本。
    final_units = qc_reference(ref_p, sr, a.chunk, model, centroid, period=period)
    suspects = hard_flags(final_units, has_target=(model is not None and centroid is not None))
    (out / "qc.json").write_text(json.dumps(suspects, ensure_ascii=False, indent=2), encoding="utf-8")
    if suspects:
        print(f"⚠️ 自检：成品里仍有 {len(suspects)} 处不合格（池子里已经换不出更好的了）：")
        for c in suspects[:12]:
            print(f"     {c['t']:5.1f}s  相似度{c['sim']:.2f} 平坦度{c['flat']:.3f} "
                  f"谱峰度{c['crest']:.0f} 峰均比{c['peak']:.1f} 基频{c['voiced']:.2f}"
                  f"  ← {' / '.join(c['why'])}")
        print(f"   → 明细 {out/'qc.json'}；这些位置对应**原片**哪一秒见 report.json")
    else:
        print("✓ 自检：成品全部通过（声纹、频谱平坦度、谱峰度、峰均比四项都在阈值内）")

    print(f"✓ 明细 → {out/'report.json'}（每块的特征值与去留原因，可人工复核）")
    print("  下一步：把这个 wav 拿去克隆音色；如果还有音效，用 report.json 里的 start 定位，"
          "加 --exclude 重跑即可。")


if __name__ == "__main__":
    main()
