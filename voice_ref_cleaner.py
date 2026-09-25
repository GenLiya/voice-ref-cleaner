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
    target_centroid: np.ndarray | None, model, chunk: float = 3.0,
) -> list[Chunk]:
    """把每个语音区间切成 <=chunk 秒的小块，逐块算特征并打分。

    打分＝各特征**排名求和**（不依赖量纲）：
      相似度 ↑、人声占比 ↑、基频稳定度 ↑、平坦度 ↓、谱峰度 ↓、峰均比 ↓
    高分的留下，低分的丢掉 —— 音效天然落在低分那一端。
    """
    out: list[Chunk] = []
    for (a, b) in regions:
        t = a
        while t + 0.3 < b:
            d = min(chunk, b - t)
            seg = voc[int(t * sr) : int((t + d) * sr)]
            if len(seg) < int(0.3 * sr):
                break
            f = dict(
                flat=_flatness(seg, sr),
                crest=_crest_ratio(seg, sr),
                peak=_peak_ratio(seg),
                voiced=_voiced_ratio(seg, sr),
                vdom=_vocal_dominance(
                    voc[int(t * sr) : int((t + d) * sr)], nov[int(t * sr) : int((t + d) * sr)], sr
                ),
                sim=float(np.dot(embed_one(seg, model), target_centroid)) if (model is not None and target_centroid is not None) else 0.0,
            )
            out.append(Chunk(start=t, dur=d, feats=f))
            t += d
    if not out:
        return out
    keys = ["sim", "vdom", "voiced", "flat", "crest", "peak"]
    higher_better = {"sim": True, "vdom": True, "voiced": True, "flat": False, "crest": False, "peak": False}
    ranks = {}
    for k in keys:
        vals = [(-c.feats[k] if higher_better[k] else c.feats[k]) for c in out]
        ranks[k] = np.argsort(np.argsort(vals))
    for i, c in enumerate(out):
        c.score = float(sum(ranks[k][i] for k in keys))
    # 归一化到 0~1（越大越好）
    smax = max(c.score for c in out)
    for c in out:
        c.score = 1.0 - c.score / (smax + EPS)
    return out


# ─────────────────────────── 主流程 ───────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="从访谈/视频音频做出干净的单人语音参考素材")
    ap.add_argument("input", help="输入音频或视频文件")
    ap.add_argument("-o", "--out", required=True, help="输出目录")
    ap.add_argument("--target", default="auto",
                    help="目标说话人：auto=自动挑；A/B/C…=指定分簇字母；或给一个参考音频路径（按声纹匹配）")
    ap.add_argument("--sample", type=float, default=40.0, help="参考素材目标时长（秒）")
    ap.add_argument("--keep-pct", type=float, default=60.0, help="按打分保留前百分之多少的块（默认 60）")
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
        regions = [r for r in regions
                   if float(np.dot(embed_one(voc16[int(r[0]*sr):int(r[1]*sr)], model), centroid)) > 0.35]
        print(f"      按声纹过滤后剩 {len(regions)} 段 / {sum(b-a_ for a_,b in regions):.1f}s")
    chunks = screen_chunks(voc16, nov16, sr, regions, centroid, model)
    chunks = [c for c in chunks if int(c.start) not in exclude]
    if not chunks:
        raise SystemExit("✗ 没有可用片段")
    order = sorted(chunks, key=lambda c: -c.score)
    n_keep = max(1, int(len(order) * a.keep_pct / 100))
    for c in order[:n_keep]:
        c.kept, c.reason = True, "保留"
    for c in order[n_keep:]:
        c.kept, c.reason = False, "打分低（疑似音效/非目标说话人）"

    gap = np.zeros(int(0.25 * sr), dtype=np.float32)
    pieces, total = [], 0.0
    for c in sorted([c for c in chunks if c.kept], key=lambda c: c.start):
        if total >= a.sample:
            break
        seg = voc16[int(c.start * sr) : int((c.start + c.dur) * sr)]
        pieces.append(seg); total += len(seg) / sr
    ref = np.concatenate([np.concatenate([p, gap]) for p in pieces])
    ref_p = out / "reference_clean.wav"
    sf.write(ref_p, ref, sr)

    report = dict(
        input=str(inp), out=str(out), demucs=not a.no_demucs,
        vad_regions=len(regions), chunks_total=len(chunks), chunks_kept=n_keep,
        target=f"{chr(65+target_label)}" if target_label is not None else a.target,
        reference_seconds=round(len(ref) / sr, 2),
        chunks=[dict(start=round(c.start, 2), dur=round(c.dur, 2), score=round(c.score, 3),
                     kept=c.kept, reason=c.reason, **{k: round(v, 4) for k, v in c.feats.items()})
                for c in sorted(chunks, key=lambda c: c.start)],
    )
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✓ 参考素材 → {ref_p}  {len(ref)/sr:.1f}s（{n_keep}/{len(chunks)} 块）")
    print(f"✓ 明细 → {out/'report.json'}（每块的特征值与去留原因，可人工复核）")
    print("  下一步：把这个 wav 拿去克隆音色；如果还有音效，用 report.json 里的 start 定位，"
          "加 --exclude 重跑即可。")


if __name__ == "__main__":
    main()
