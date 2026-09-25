# voice-ref-cleaner

**从访谈 / 视频音频里，做出"干净的单人语音参考素材"，供音色克隆（voice cloning）与 TTS 使用。**

> Turn a messy interview/video track into a clean single-speaker reference clip for voice cloning:
> **remove BGM → isolate the target speaker → strip sound effects.**

---

## ⚠️ 出处与致谢（请先读这段）

这个项目**不是从零造的**，它站在三个成熟开源项目上。**前面两个是别人的，本项目只调用、不修改**：

| 组件 | 作用 | 出处 | 许可 |
|---|---|---|---|
| **Demucs** | **去 BGM**：音源分离，把人声与伴奏拆开 | <https://github.com/facebookresearch/demucs> — Alexandre Défossez 等（Meta AI / INRIA）。论文：*Hybrid Transformers for Music Source Separation* (ICASSP 2023)；*Music Source Separation in the Waveform Domain* (2019) | MIT |
| **CAM++ / 3D-Speaker** | **按说话人分离**：声纹向量 | <https://github.com/modelscope/3D-Speaker>（Apache-2.0）；权重 `iic/speech_campplus_sv_zh-cn_16k-common`（ModelScope，阿里达摩院） | Apache-2.0 |
| **Silero VAD** | 语音区间检测 | <https://github.com/snakers4/silero-vad> | MIT |

**去 BGM 这件事完全是 Demucs 做的**——如果你的需求只是"去掉背景音乐"，请直接去用 Demucs，
不需要这个项目。本项目的价值在下面这两步。

---

## 本项目加了什么

只做 Demucs 是不够的。实测：Demucs 的人声轨里**仍然残留大量音效瞬态**（"叮/咚/噔/咻"、
"嘿"这类人声喊叫、以及别人的插话），而这些一旦进了克隆参考素材，**克隆出来的音色就会被污染**。

本项目在 Demucs 之后加了三步：

1. **按说话人分离（CAM++ 聚类）**
   访谈类音频往往是"主持人提问 + 目标人回答"。整段拿去克隆 = 把两个人的音色混成一个。
   实测一条 400 秒的访谈里分出 **3 个说话人**，而按"画面里嘴在动"这条客观判据，
   真正的目标说话人只占 **54.6%** —— 也就是说**近一半素材是别人的声音**。

2. **VAD 按句切分，而不是在固定 1 秒网格上切**
   在网格上切，一个音效会把整块 1 秒都污染；按语音区间切、并按句边界分块，音效落在停顿里就自然被丢掉。

3. **逐块音效筛查（六个特征排名求和）**

   | 特征 | 直觉 | 方向 |
   |---|---|---|
   | `sim` 声纹相似度（与目标说话人质心） | 音效/别人的声音偏低 | 越高越好 |
   | `vdom` **人声轨 / 伴奏轨能量比** | 用 Demucs 的两条输出互相比 —— 音效与音乐残留的片段，伴奏能量更高 | 越高越好 |
   | `voiced` 稳定基频帧占比 | 音效没有基频 | 越高越好 |
   | `flat` 频谱平坦度 | 语音有谐波结构（低），宽带噪声（高） | 越低越好 |
   | `crest` 谱峰度 | "叮、噔"这类窄带强音会很高 | 越低越好 |
   | `peak` 峰均比 | "嘿"这类爆音很尖 | 越低越好 |

   六个特征**各自排名后求和**（不依赖量纲），取分数高的一段。
   实测保留块与淘汰块的分布确实分得开：

   | | 人声占比 | 峰均比 | 基频占比 |
   |---|---|---|---|
   | 保留（前 60%） | **15.8 ~ 27.5 dB** | **1.9 ~ 3.2** | **0.93 ~ 0.99** |
   | 淘汰（后 40%） | 7.8 ~ 14.5 dB | 4.0 ~ 12.4 | 0.67 ~ 0.87 |

另外还有两个小改进：
- **簇编号按"首次出现时间"固定** —— KMeans 的簇编号是随机的，同一段音频跑两次同一个说话人
  可能这次叫 A、下次叫 C；用户是按字母反馈的，编号一变反馈就错位。
- **输出可复核的 `report.json`**：每一块的特征值与去留原因都在里面，
  人工发现漏网的音效时，可以直接用 `--exclude` 点名秒数重跑。

---

## 安装

```bash
# 系统依赖：ffmpeg（必须在 PATH 里）
pip install -r requirements.txt              # 核心：Demucs + VAD + 特征筛查
pip install -r requirements-speaker.txt      # 可选：说话人分离（ModelScope CAM++）
```

首次运行会下载 Demucs 的 `htdemucs` 权重（约 80 MB）。CAM++ 权重约 27 MB，随 ModelScope 下载。

---

## 用法

```bash
python voice_ref_cleaner.py 访谈.mp4 -o out/
```

跑完会得到：

```
out/
├── speaker_A.wav        ← 每个说话人一段样本，听这个指认目标
├── speaker_B.wav
├── speaker_C.wav
├── reference_clean.wav  ← ★ 最终交付：拿去克隆音色的参考素材
└── report.json          ← 每块的特征值与去留原因
```

**第一步一定要先听 `speaker_*.wav`**，确认哪一个是你要的人，然后重跑并指定：

```bash
python voice_ref_cleaner.py 访谈.mp4 -o out/ --target B          # 指定分簇字母
python voice_ref_cleaner.py 访谈.mp4 -o out/ --target 样本.wav   # 或用一段参考音频按声纹匹配
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--sample 40` | 参考素材目标时长（秒），默认 40 |
| `--keep-pct 55` | 按打分保留前百分之多少的块，默认 60；音效多就调低 |
| `--exclude "45,63,219"` | **人工点名**要跳过的源音频秒数（配合 report.json 用） |
| `--no-demucs` | 输入已经是干净人声时跳过 |
| `--no-speaker` | 只有一个人说话时跳过说话人分离 |
| `--device cuda` | 有 N 卡时 Demucs 跑 GPU（快很多） |

Demucs 的结果会缓存在 `out/demucs/`，**反复调参不会重跑分离**（CPU 上那是几分钟的事）。

---

## 它内部干了什么

```
输入(音/视频)
   │  ffmpeg 抽 44.1k
   ├─▶ ① Demucs --two-stems=vocals ──▶ vocals.wav / no_vocals.wav
   │                                    （两条都留着，后面要用它们的能量比）
   ├─▶ ② Silero VAD ──────────────────▶ 语音区间列表
   ├─▶ ③ CAM++ 滑窗向量 → KMeans ─────▶ 每簇样本（人工指认）→ 目标簇质心
   ├─▶ ④ 按句切块 → 6 特征打分 → 取高分
   └─▶ reference_clean.wav + report.json
```

---

## 边界与不保证

- **去不干净是常态。** 音效与人声在时频域重叠时，任何分离器都无能为力；
  这个工具的策略是"**有 100 块候选、只要 40 块 → 只挑最像正常说话的**"，而不是"把音效从波形里抠掉"。
  所以它**会丢掉一些本来能用的素材**，这是刻意的取舍。
- **不需要 GPU**，但 CPU 上 Demucs 大约是实时的 1.5 倍耗时（400 秒音频 ≈ 3 分钟）。
- **不做声纹的"身份认定"**：只做"两段像不像"和聚类，不要用于任何身份核验场景。
- 请自行确认你对输入素材拥有相应权利。**用真人声音做克隆并公开传播，涉及肖像权与声音权**
  （中国法下见民法典第 1019 / 1023 条），平台另有 AI 生成内容标识要求。

---

## License

MIT（本项目自身代码）。**上游组件各自遵循其原有许可**：Demucs(MIT)、Silero VAD(MIT)、
3D-Speaker / CAM++(Apache-2.0)。使用前请一并遵守。
