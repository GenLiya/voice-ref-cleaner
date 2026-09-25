# 第三方组件与致谢

**本项目的核心能力里，"去 BGM" 这一步完全是别人的工作。** 我们不修改这些项目，只调用它们。
如果你只是想"去掉背景音乐"，请直接去用 Demucs，不需要这个项目。

| 组件 | 在本项目里的作用 | 出处与作者 | 许可 |
|---|---|---|---|
| **Demucs** | **去 BGM**：音源分离，把人声与伴奏拆开（`--two-stems=vocals`，模型 `htdemucs`） | <https://github.com/facebookresearch/demucs><br>Alexandre Défossez 等（Meta AI / INRIA）<br>论文：*Hybrid Transformers for Music Source Separation*, ICASSP 2023；*Music Source Separation in the Waveform Domain*, 2019 | MIT |
| **CAM++ / 3D-Speaker** | **按说话人分离**：192 维声纹向量与聚类<br>权重 `iic/speech_campplus_sv_zh-cn_16k-common` | <https://github.com/modelscope/3D-Speaker><br>ModelScope / 阿里达摩院 | Apache-2.0 |
| **Silero VAD** | 语音区间检测（按句切分） | <https://github.com/snakers4/silero-vad><br>Silero Team | MIT |
| **librosa / SoundFile / scikit-learn / PyTorch** | 重采样、读写音频、KMeans 与轮廓系数、张量运算 | 各自项目主页 | ISC / BSD / BSD / BSD |

## 对 Demucs 的具体引用方式

本项目**不包含也不修改 Demucs 的代码**，而是以子进程方式调用它的官方命令行：

```bash
python -m demucs --two-stems=vocals -n htdemucs -d cpu -o <outdir> <input>
```

输出 `vocals.wav` 与 `no_vocals.wav`。后者本不用丢 —— 本项目会拿两条轨的**能量比**
作为"这一段到底是人在说话、还是音效/音乐残留"的判据之一（代码里的 `vocal_dominance`）。

## 对 CAM++ 的具体引用方式

同样不包含模型代码，通过 ModelScope 的 pipeline 加载：

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
sv = pipeline(task=Tasks.speaker_verification, model='iic/speech_campplus_sv_zh-cn_16k-common')
emb = sv.model(waveform_tensor)     # → 192 维声纹向量
```

> 说明：ModelScope 的官方文档只演示了"比较两条音频像不像"（`sv([a, b])`），
> 而做聚类/归属需要**向量**本身。上面这种直接取 `sv.model(波形)` 的用法是本项目摸出来的。

## 许可与责任

本项目自身代码以 MIT 发布（见 `LICENSE`）。**上游组件各自遵循其原有许可**，使用前请一并遵守。
模型权重的使用条款以各模型主页为准。

使用者需自行确认对输入素材拥有相应权利。**用真人声音做克隆并公开传播，涉及肖像权与声音权**
（中国法下见《民法典》第 1019 条、第 1023 条），各平台另有 AI 生成内容的标识要求。
