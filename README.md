# ZerolanCore

![Static Badge](https://img.shields.io/badge/Python-3.10-blue) ![Static Badge](https://img.shields.io/badge/LLM-purple) ![Static Badge](https://img.shields.io/badge/ASR-purple) ![Static Badge](https://img.shields.io/badge/TTS-purple) ![Static Badge](https://img.shields.io/badge/OCR-purple) ![Static Badge](https://img.shields.io/badge/Image%20Captioning-purple) ![Static Badge](https://img.shields.io/badge/Video%20Captioning-purple) ![Static Badge](https://img.shields.io/badge/License-MIT-orange) ![Static Badge](https://img.shields.io/badge/ver-1.2-green) 

ZerolanCore 集成了众多开源的、可本地部署的人工智能模型或服务，旨在使用统一的管线设计封装大语言模型（LLM）、自动语音识别（ASR）、文本转语音（TTS）、图像字幕（Image Captioning）、光学字符识别（OCR）、视频字幕（Video Captioning）等一系列的人工智能模型，并可以使用统一的配置文件和服务启动器快速部署和启动 AI 服务。

>  相关项目：[ZerolanLiveRobot](https://github.com/AkagawaTsurunaki/ZerolanLiveRobot)、[ZerolanData](https://github.com/AkagawaTsurunaki/zerolan-data)

## 项目核心结构

本项目的核心模块结构如下，你可以根据需要选择安装不同类型的 AI 模型：

```
├─ asr		# 自动语音识别模型
├─ img_cap  # 图像字幕模型
├─ ...
└─ llm		# 大语言模型
    ├─ app.py		# 统一的应用封装，每个 AI 模型模块都由它的父模块中的这个 app.py 以 Web 服务器的形式加载并启动
    └─ paraformer	# 指定模型名称（通常是简写）
       ├─ config.py           # 该模型的配置文件
       ├─ model.py			  # 封装了该模型，并遵循了统一的 Pipeline API（意味着同一类模型的对外接口统一，即便官方实现的接口各有不同）
       ├─ pyproject.toml	  # uv 生成的项目配置清单（若存在，强烈推荐使用这个）
       ├─ uv.lock       	  # uv 生成的依赖配置清单，严格记录了这个模型的依赖项（若存在，强烈推荐使用这个）
       └─ requirements.txt    # 运行该模型需要的 Python 依赖
```

## 模型配置文件

将项目根目录中的配置文件 `config.template.yaml` 复制一份并更名为 `config.yaml`，然后修改之中的配置项。

以 `LLM` 配置项为例。

### 选择模型

如果你想选择 `Qwen/Qwen-7B-Chat` 作为你想要部署的大语言模型，那么需要在 `id` 项填写模型名称 `Qwen/Qwen-7B-Chat`，模型名称在下一节的各个表格的第一列（严格匹配）。

### 设置服务 IP 地址

然后设置 `host` 项，这是模型 Web 服务的主机地址，默认是 `0.0.0.0`，这意味着该模型的服务会监听所有可用的网络接口，允许从任意 IP 地址访问该服务。
如果将 `host` 设置 `127.0.0.1`，那么仅有本机可以访问这个服务。当然，你也可以指定 IP 地址，例如 `192.168.2.1`。如果你选择公网访问，别忘记合理配置防火墙的放行规则。

> 至于选择哪个，取决于你将
> [ZerolanLiveRobot](https://github.com/AkagawaTsurunaki/ZerolanLiveRobot) 与 [ZerolanCore](https://github.com/AkagawaTsurunaki/zerolan-core) 
> 部署在什么样的环境中。
> 
> 例如你将 [ZerolanCore](https://github.com/AkagawaTsurunaki/zerolan-core) 
> 放到一个你购买的远程服务器上，而 [ZerolanLiveRobot](https://github.com/AkagawaTsurunaki/ZerolanLiveRobot) 放到你的笔记本电脑上（和远程服务器不是一个主机），
> 此时就必须设置为 `0.0.0.0`；否则，如果 [ZerolanLiveRobot](https://github.com/AkagawaTsurunaki/ZerolanLiveRobot) 和 [ZerolanCore](https://github.com/AkagawaTsurunaki/zerolan-core)
> 完全在同一台电脑上，设置 `0.0.0.0` 或 `127.0.0.1` 均可，但 `127.0.0.1` 在程序上更安全。

接着设置 `port` 项，这是模型 Web 服务的端口号，推荐你使用默认的端口号，但是如果端口号与其他进程冲突则需要更换。

### 配置模型设置

在配置文件中，可以看到这样的结构：

```yaml
LLM:
  config:
    ...
    Qwen/Qwen-7B-Chat:
      model_path: "your/path/to/Qwen-7B-Chat" # Directory of the model
      ...
```

因为你选择了 `Qwen/Qwen-7B-Chat` 所以只需要修改你启动的那个模型的配置就行了（不启用的模型不用管）。

`model_path` 是模型的地址，严格来说是一个路径。默认配置下，会尝试从 HuggingFace 模型仓库下载模型，由于部分地区连接 HuggingFace 存在困难，你可以选择配置环境变量。

Linux 设置环境变量：

```shell
export HF_ENDPOINT=https://hf-mirror.com
```

Windows 设置环境变量：

```shell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

如果使用镜像也无法下载，您可能需要自行探究方法手动下载模型并设置模型地址（目录）。

剩下的选项因模型而异，详细可以看配置文件中的内容，如果你不知道怎么改，就保持默认配置，因为这些配置经过了测试确实可用。

## 支持集成模型

以下的模型已经集成在 ZerolanCore 中，并均过作者在 Windows 11 和 Ubuntu 22.04 两个主流系统上进行了测试，可以正常使用。然而不同系统的环境差异显著，实在无法广泛覆盖所有情况，如有意外敬请谅解。

> [!CAUTION]
>
> 如果运行的模型所需要显存大小，远远超过你的系统的显存与内存之和，这可能造成**系统崩溃**。
>
> 因此在模型加载的过程中，请时刻留意你的系统资源状况。在 Windows 中，使用 `CTRL` + `SHIFT` + `ECS` 调出任务管理器进行监视；Ubuntu 上可以使用`nvtop` 监视显存占用，使用 `top` 监视内存占用。
>
> 需要注意的是，以下表格中的显存占用数据是，模型启动后执行一次回复后的显存占用结果，但是随着推理的进行，可能会出现显存占用稍微增加的现象，如果您的显卡显存较小，请尤其注意。

### 大语言模型

根据自然语言上下文进行推理，遵循用户指令，并给予文字响应。

| 模型名称                                                                 | 支持语言 | 流式推理 | 显存占用                                                |
|----------------------------------------------------------------------|------| -------- |-----------------------------------------------------|
| [THUDM/GLM-4](https://github.com/THUDM/GLM-4)                        | 中英   |     ❌️     | 18.4 GiB                                            |
| [THUDM/chatglm3-6b](https://github.com/THUDM/ChatGLM3)               | 中英   | ✅️        | 无量化 12.4 GiB \| 8-Bit 量化 7.5  GiB \| 4-Bit 量化 4.6 GiB |
| [Qwen/Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat)        | 中英   | ✅️        | 15.3 GiB                                            |
| [01-ai/Yi-6B-Chat](https://www.modelscope.cn/models/01ai/Yi-6B-Chat) | 中英   | ❌️        | 23.7 GiB                                            |
| [augmxnt/shisa-7b-v1](https://huggingface.co/augmxnt/shisa-7b-v1)    | 日英   | ❌️        | 16.0 GiB                                            |
| [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) | 中英   |❌️| 47.2 GiB                                            |
| [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) | 中英   |❌️| 48.0 GiB 以上                                           |  

> [!NOTE]
>
> 1. [THUDM/chatglm3-6b](https://github.com/THUDM/ChatGLM3) 偶尔存在**中文夹杂英文**的现象，且量化精度越低这种现象越严重。
> 2. [THUDM/GLM-4](https://github.com/THUDM/GLM-4)  在工具调用时返回的 JSON 字符串高概率存在 **JSON 语法错误**。
> 3. [Qwen/Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat) 测试时发现使用多卡推理可能会**报错**，因此您应该使用**单卡推理**。
> 4. [augmxnt/shisa-7b-v1](https://huggingface.co/augmxnt/shisa-7b-v1) 在测试时可能发生无法读取上下文的问题。
> 5. [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) 支持双卡推理，但是存在语句异常中断问题，原因不详。
> 6. [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) 支持双卡推理，但是其显存已经超出两张 NVIDIA GeForce RTX 4090 的极限。

---
使用以下命令创建 [THUDM/GLM-4](https://github.com/THUDM/GLM-4) 的环境并启动模型。

如果使用 `uv`，运行：

```shell
cd llm/glm4
uv sync
source .venv/bin/activate
cd ../../
uv run starter.py llm
```

如果使用 `Anaconda`，运行：

```shell
cd llm/glm4
conda create --name llm_glm4 python==3.11 --yes
conda activate llm_glm4
pip install -e .
cd ../../
python starter.py llm
```
---

使用以下命令创建 [THUDM/chatglm3-6b](https://github.com/THUDM/ChatGLM3) 的环境并启动模型。

如果使用 `uv`，运行：

```shell
cd llm/chatglm3
uv sync
source .venv/bin/activate
cd ../../
uv run starter.py llm
```

如果使用 `Anaconda`，运行：

```shell
cd llm/chatglm3
conda create --name llm_chatglm3 python==3.10 --yes
conda activate llm_chatglm3
pip install -e .
cd ../../
python starter.py llm
```

---

使用以下命令创建 [Qwen/Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat) 的环境并启动模型。

如果使用 `uv`，运行：

```shell
cd llm/qwen
uv sync
source .venv/bin/activate
cd ../../
uv run starter.py llm
```

如果使用 `Anaconda`，运行：

```shell
cd llm/qwen
conda create --name llm_qwen python==3.11 --yes
conda activate llm_qwen
pip install -e .
cd ../../
python starter.py llm
```

---

使用以下命令创建 [01-ai/Yi-6B-Chat](https://www.modelscope.cn/models/01ai/Yi-6B-Chat) 的环境并启动模型。

如果使用 `uv`，运行：

```shell
cd llm/yi
uv sync
source .venv/bin/activate
cd ../../
uv run starter.py llm
```

如果使用 `Anaconda`，运行：

```shell
cd llm/yi
conda create --name llm_yi python==3.11 --yes
conda activate llm_yi
pip install -e .
cd ../../
python starter.py llm
```

---

使用以下命令创建 [augmxnt/shisa-7b-v1](https://huggingface.co/augmxnt/shisa-7b-v1) 的环境并启动模型。

如果使用 `uv`，运行：

```shell
cd llm/shisa
uv sync
source .venv/bin/activate
cd ../../
uv run starter.py llm
```

如果使用 `Anaconda`，运行：

```shell
cd llm/shisa
conda create --name llm_shisa python==3.11 --yes
conda activate llm_shisa
pip install -e .
cd ../../
python starter.py llm
```

---

使用此命令创建 [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)、[DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) 的环境并启动模型。

如果使用 `uv`，运行：

```shell
cd llm/deepseek
uv sync
source .venv/bin/activate
cd ../../
uv run starter.py llm
```

如果使用 `Anaconda`，运行：

```shell
cd llm/deepseek
conda create --name llm_deepseek python==3.11 --yes
conda activate llm_deepseek
pip install -e .
cd ../../
python starter.py llm
```

---

测试大语言模型的文字回复功能是否正常：

```shell
curl -X POST http://localhost:11002/llm/predict \
-H "Content-Type: application/json; charset=utf-8" \
-d @- <<EOF
{
    "text": "What is my name?",
    "history": [
        {"content": "You are a helpful assistant!", "metadata":null, "role":"system"},
        {"content": "My name is AkagawaTsurunaki.", "metadata":null, "role":"user"},
        {"content": "Hello, AkagawaTsurunaki.", "metadata":null, "role":"assistant"}
    ]
}
EOF
```

返回值应该类似：

```json
{"id":"f57f9f9c-7109-4459-8bf3-48f7e5e4597c","response":"\nYour name is AkagawaTsurunaki. It's quite unique!","history":[{"role":"system","content":"You are a helpful assistant!","metadata":null},{"role":"user","content":"My name is AkagawaTsurunaki.","metadata":null},{"role":"assistant","content":"Hello, AkagawaTsurunaki.","metadata":null},{"role":"assistant","content":"\nYour name is AkagawaTsurunaki. It's quite unique!","metadata":null}]}
```

### 自动语音识别模型

识别一段自然语言语音，将其内容转换为文本字符串。

| 模型名称                                                                                                                                                                        | 支持语言 | 显存占用    |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|---------|
| [iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1](https://www.modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1) | 中英  | 0.5 GiB |
| [kotoba-tech/kotoba-whisper-v2.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0) | 日   | 2.2 GiB |

> [!NOTE]
>
> 1. [iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1](https://www.modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1) 在本项目没有使用**符号分割**和**音频激活**子模型，如有需要请[查看此处](https://www.modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1)。

---

使用以下命令创建 [iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1](https://www.modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1) 的环境并启动模型。

如果使用 `uv`，运行：

```shell
cd asr/paraformer
uv sync
source .venv/bin/activate
cd ../../
uv run starter.py asr
```

如果使用 `Anaconda`，运行：

```shell
cd asr/paraformer
conda create --name asr_paraformer python==3.11 --yes
conda activate asr_paraformer
pip install -e .
cd ../../
python starter.py asr
```

---

使用以下命令创建 [kotoba-tech/kotoba-whisper-v2.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0) 的环境并启动模型。

如果使用 `uv`，运行：

```shell
cd asr/kotoba_whisper_2
uv sync
source .venv/bin/activate
cd ../../
uv run starter.py asr
```

如果使用 `Anaconda`，运行：

```shell
cd asr/kotoba_whisper_2
conda create --name asr_kotoba_whisper_2 python==3.11 --yes
conda activate asr_kotoba_whisper_2
pip install -e .
cd ../../
python starter.py asr
```

---

测试模型的语音识别是否正常（注意需要从项目所在目录作为当前工作目录执行）：

```shell
curl -X POST http://localhost:11001/asr/predict \
  -F "audio=@tests/resources/tts-test.wav;type=audio/wav" \
  -F "json={\"audio_path\": \"\", \"channels\": 2};type=application/json"
```

返回值应该类似：

```json
{"id":"b5a1dbaf-6e32-4750-90b4-0dce0294a4fe","transcript":"我是赤川鹤鸣"}
```

### 文本转语音模型

根据给定的参考音频和文本，生成对应的语音。

| 模型名称                                                     | 支持语言   | 流式推理 | 显存占用 |
| ------------------------------------------------------------ | ---------- | -------- | -------- |
| [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | 中粤日英韩 |    ✅️      | 1.3 GiB  |

> [!IMPORTANT]
> 
> 1. [GPT-SoVITS](https://github.com/AkagawaTsurunaki/GPT-SoVITS) 的安装教程请参考官方 `README.md`，请注意必须是[此 Forked 版本](https://github.com/AkagawaTsurunaki/GPT-SoVITS)才能与本项目的接口适配。**不要使用官方的整合包，因为接口实现与本项目不匹配。**

关于 GPT-SoVITS 详细的启动方法如下。

首先将项目克隆下来，切换到 `zerolan` 分支

```shell
git clone https://github.com/AkagawaTsurunaki/GPT-SoVITS.git
cd GPT-SoVITS
# 假设你已经按照 GPT-SoVITS 官方的 README.md 配置好了环境（步骤比较多，请保持耐心）
python zerolan_api.py -a 127.0.0.1 -p 11004
```

需要下载 nltk_data

### 图像字幕模型

识别一张图片，生成对这张图片内容的文字描述。用于为大语言模型提供视觉信息，但近年来的多模态模型已经开始发展，此类模型在未来可能会被弃用。

| 模型名称                                                                                                    | 支持语言 | 显存占用    |
|---------------------------------------------------------------------------------------------------------|------|---------|
| [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large) | 英文   | 1.1 GiB |

> [!NOTE]
>
> 1. [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large) 存在一定的幻觉问题，即容易生成与图片中内容无关的内容。

---

使用以下命令创建 [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large) 的环境并启动模型。

如果使用 `uv`，运行：

```shell
cd img_cap/blip
uv sync
source .venv/bin/activate
cd ../../
uv run starter.py imgcap
```

如果使用 `Anaconda`，运行：

```shell
cd img_cap/blip
conda create --name img_cap_blip python==3.11 --yes
conda activate img_cap_blip
pip install -e .
cd ../../
python starter.py imgcap
```

---

测试图像字幕模型的图像语义识别功能是否正常（注意需要从项目所在目录作为当前工作目录执行）：

```shell
curl -X POST http://localhost:11003/img-cap/predict \
-H "Content-Type: application/json; charset=utf-8" \
-d @- <<EOF
{
    "img_path": "./tests/resources/imgcap-test.png"
}
EOF
```

返回值应该类似：

```json
{"caption":"there is a cartoon girl with a ponytail and a red dress","id":"a9c09210-63f7-4b2b-9119-50ab116cadf3","lang":"en"}
```

### 视频字幕模型

| 模型名称                                                     | 支持语言 | 流式推理 | 显存占用 |
| ------------------------------------------------------------ | -------- | -------- | ---- |
| [iic/multi-modal_hitea_video-captioning_base_en](https://www.modelscope.cn/models/iic/multi-modal_hitea_video-captioning_base_en) | 英       |    ❌️      | 尚未测试 |

### 光学字符识别模型

识别一张图片，并将其中包含的文字字符提取出。

| 模型名称                                                               | 支持语言   | 显存占用    |
|--------------------------------------------------------------------|--------|---------|
| [paddlepaddle/PaddleOCR](https://gitee.com/paddlepaddle/PaddleOCR) | 中英法德韩日 | 0.5 GiB |

---

使用以下命令创建 [paddlepaddle/PaddleOCR](https://gitee.com/paddlepaddle/PaddleOCR) 的环境并启动模型。

如果使用 `uv`，运行：

```shell
cd ocr/paddle
uv sync
source .venv/bin/activate
cd ../../
uv run starter.py ocr
```

如果使用 `Anaconda`，运行：

```shell
cd ocr/paddle
conda create --name ocr_paddle python==3.11 --yes
conda activate ocr_paddle
pip install -e .
cd ../../
python starter.py ocr
```

---

测试光学字符识别模型的文字识别功能是否正常（注意需要从项目所在目录作为当前工作目录执行）：

```shell
curl -X POST http://localhost:11004/ocr/predict \
-H "Content-Type: application/json; charset=utf-8" \
-d @- <<EOF
{
    "img_path": "./tests/resources/ocr-test.png"
}
EOF
```

返回值应该类似：

```json
{"id":"83316895-f0d9-4d84-adfa-1acf5d02354f","region_results":[{"position":{"lu":{"x":8.0,"y":25.0},"ru":{"x":434.0,"y":25.0},"rd":{"x":434.0,"y":87.0},"ld":{"x":8.0,"y":87.0}},"content":"我是赤川鹤鸣","confidence":0.9613597989082336}]}
```

### 视觉语言模型代理

根据图片的内容以及用户文本指令的指导，执行某种动作。

| 模型名称                             | 支持语言   | 显存占用    |
|----------------------------------------------------|--------|---------|
| [showlab/ShowUI](https://github.com/showlab/ShowUI) | 中英 | 10.9 GiB |

1. [showlab/ShowUI](https://github.com/showlab/ShowUI) 可以在用户指令和给定图片中模拟人类操作 UI 界面给予动作反馈。例如你可以使用“点击搜索按钮”。

## License

Feel free to enjoy open-source!

[MIT License](https://github.com/AkagawaTsurunaki/zerolan-core/blob/dev/LICENSE)

## Contact with me

Email: AkagawaTsurunaki@outlook.com

Github: [AkagawaTsurunaki](https://github.com/AkagawaTsurunaki)

Bilibili: [赤川鹤鸣_Channel](https://space.bilibili.com/1076299680)
