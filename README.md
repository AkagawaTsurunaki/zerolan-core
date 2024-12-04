# ZerolanCore

![Static Badge](https://img.shields.io/badge/Python-3.10-blue) ![Static Badge](https://img.shields.io/badge/Docker-Supported-blue) ![Static Badge](https://img.shields.io/badge/LLM-purple) ![Static Badge](https://img.shields.io/badge/ASR-purple) ![Static Badge](https://img.shields.io/badge/TTS-purple) ![Static Badge](https://img.shields.io/badge/OCR-purple) ![Static Badge](https://img.shields.io/badge/Image%20Captioning-purple) ![Static Badge](https://img.shields.io/badge/Video%20Captioning-purple) ![Static Badge](https://img.shields.io/badge/License-MIT-orange) ![Static Badge](https://img.shields.io/badge/ver-1.1-green) 

ZerolanCore 集成了众多开源的、可本地部署的人工智能模型或服务，旨在使用统一的管线设计封装大语言模型（LLM）、自动语音识别（ASR）、文本转语音（TTS）、图像字幕（Image Captioning）、光学字符识别（OCR）、视频字幕（Video Captioning）等一系列的人工智能模型，并可以使用统一的配置文件和服务启动器快速部署和启动 AI 服务。

>  相关项目：[ZerolanLiveRobot](https://github.com/AkagawaTsurunaki/ZerolanLiveRobot)、[ZerolanData](https://github.com/AkagawaTsurunaki/zerolan-data)

## 部署准备工作

假设你已经具有足够显存的系统，并安装了 Python、Anaconda（可选）和 Docker（可选），让我们开始吧。

### 项目核心结构

本项目的核心模块结构如下，你可以根据需要选择安装不同类型的 AI 模型：

```
├─ llm		# 大语言模型
├─ img_cap  # 图像字幕模型
├─ ...
└─ asr		# 自动语音识别模型
    ├─ app.py		# 统一的应用封装，每个 AI 模型模块都由它的父模块中的这个 app.py 以 Web 服务器的形式加载并启动
    └─ paraformer	# 指定模型名称（通常是简写）
       ├─ config.py           # 该模型的配置文件
       ├─ Dockerfile          # 该模型的 Docker 文件，能构建支持运行本模型 Docker Container
       ├─ model.py			  # 封装了该模型，并遵循了统一的 Pipeline API（意味着同一类模型的对外接口统一，即便官方实现的接口各有不同）
       └─ requirements.txt    # 运行该模型需要的 Python 依赖
```

### 运行环境构建

以 Docker 为例，如果想要部署 [iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1](https://www.modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1)，我们需要找到 `./asr/paraformer/Dockerfile` 中的 Docker 文件，然后运行以下指令：

```shell
docker build ./asr/paraformer --tag speech_paraformer_asr
docker run -d --gpus all -it -p 11001:11001 --name my_conatainer speech_paraformer_asr
```

这将构建一个 tag 为 `speech_paraformer_asr` 的 Docker 镜像，并基于此镜像运行一个使用全部 GPU，对外暴露 11001 端口的 Docker Container，并指定这个容器的名称为 `my_conatainer`。

> [!IMPORTANT]
> 
> Docker 文件中指定了一些镜像，这是为了更快的下载和构建。如果你不需要它们，请根据实际需求删除这些指令。

如果不希望构建 Docker Container，可以在 Anaconda 中建立多个虚拟环境，或者直接使用 `pyenv` 建立虚拟环境，并使用 `pip` 进行安装。

以 Anaconda 为例，运行以下命令：

```shell
conda create --name speech_paraformer_asr python=3.10
```

这将创建一个名为 `speech_paraformer_asr` 的 conda 环境，指定 Python 版本为 3.10。

然后激活这个环境，使用：

```
conda activate speech_paraformer_asr
pip install -r requirements.txt
pip install -r ./asr/paraformer/requirements.txt
```

这将会自动下载并安装所有依赖。

>  [!IMPORTANT]
>
> 由于不同的模型所需要的环境各不相同，强烈建议为每个模型使用相互隔离的 Python 环境，以免出现依赖冲突等问题。

其他模型的运行环境构建类似，此不赘述。

### 模型配置文件

将项目根目录中的配置文件 `config.template.yaml` 更名为 `config.yaml`，然后修改之中的配置项，详细内容请看配置文件中的注释内容。

> [!NOTE]
> 
> 如果在 `ip` 中设置 `127.0.0.1`，那么仅有本机可以访问这个服务。

> [!IMPORTANT]
> 
> 默认配置下，ZerolanCore 会尝试从 Hugging Face 下载部分模型，由于部分地区连接 Hugging Face 存在困难，您可能需要手动下载模型并设置模型地址。

### 启动模型服务

如果一切顺利，你将使用这条命令启动自动语音识别（ASR）服务：

```
python starter.py asr
```

你也可以使用参数 `llm`、`img_cap`、`ocr`、`vid_cap`、`tts` 等，详见 `starter.py`。

如果你的终端上没有报错，且看到了模型的加载进度条，且有类似网络 IP 的字样，则可视为启动成功。

其他服务的启动类似，此不赘述。

## 支持集成模型

以下的模型已经集成在 ZerolanCore 中，并均过作者在 Windows 11 和 Ubuntu 22.04 两个主流系统上进行了测试，可以正常使用。然而不同系统的环境差异显著，实在无法广泛覆盖所有情况，如有意外敬请谅解。

> [!CAUTION]
>
> 如果运行的模型所需要显存大小，远远超过你的系统的显存与内存之和，这可能造成**系统崩溃**。
>
> 因此在模型加载的过程中，请时刻留意你的系统资源状况。在 Windows 中，使用 `CTRL` + `SHIFT` + `ECS` 调出任务管理器进行监视；Ubuntu 上可以使用`nvtop` 监视显存占用，使用 `top` 监视内存占用。

### 大语言模型

根据自然语言上下文进行推理，遵循用户指令，并给予文字响应。

| 模型名称                                                     | 支持语言 | 流式推理 | 显存占用                                                     |
| ------------------------------------------------------------ | -------- | -------- | ------------------------------------------------------------ |
| [THUDM/GLM-4](https://github.com/THUDM/GLM-4)                | 中英     |     ❌️     | 19.2 GiB                                                     |
| [THUDM/chatglm3-6b](https://github.com/THUDM/ChatGLM3)       | 中英     | ✅️        | 无量化 11.5 GiB \| 8-Bit 量化 6.6  GiB \| 4-Bit 量化 4.0 GiB |
| [Qwen/Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat) | 中英     | ✅️        | 11.5 GiB                                                     |
| [01-ai/Yi-6B-Chat](https://www.modelscope.cn/models/01ai/Yi-6B-Chat) | 中英     | ❌️        | 10.0 GiB                                                     |
| [augmxnt/shisa-7b-v1](https://huggingface.co/augmxnt/shisa-7b-v1) | 日英     | ❌️        | 11.4 GiB                                                     |

> [!NOTE]
>
> 1. [THUDM/chatglm3-6b](https://github.com/THUDM/ChatGLM3) 偶尔存在**中文夹杂英文**的现象，且量化精度越低这种现象越严重。
> 2. [THUDM/GLM-4](https://github.com/THUDM/GLM-4)  在工具调用时返回的 JSON 字符串高概率存在 **JSON 语法错误**。
> 3. [Qwen/Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat) 测试时发现使用多卡推理可能会**报错**，因此您应该使用**单卡推理**。

### 自动语音识别模型

识别一段自然语言语音，将其内容转换为文本字符串。

| 模型名称                                                                                                                                                                          | 支持语言 | 显存占用    |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|---------|
| [iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1](https://www.modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1) | 中英   | 0.5 GiB |

> [!NOTE]
>
> 1. [iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1](https://www.modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1) 在本项目没有使用**符号分割**和**音频激活**子模型，如有需要请[查看此处](https://www.modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1)。

### 文本转语音模型

根据给定的参考音频和文本，生成对应的语音。

| 模型名称                                                     | 支持语言   | 流式推理 | 显存占用 |
| ------------------------------------------------------------ | ---------- | -------- | -------- |
| [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | 中粤日英韩 |    ✅️      | 1.3 GiB  |

> [!IMPORTANT]
> 
> 1. [GPT-SoVITS](https://github.com/AkagawaTsurunaki/GPT-SoVITS) 的安装教程请参考官方 `README.md`，请注意必须是[此 Forked 版本](https://github.com/AkagawaTsurunaki/GPT-SoVITS)才能与本项目的接口适配。**不要使用官方的整合包，因为接口实现与本项目不匹配。**

### 图像字幕模型

识别一张图片，生成对这张图片内容的文字描述。

| 模型名称                                                                                                    | 支持语言 | 显存占用    |
|---------------------------------------------------------------------------------------------------------|------|---------|
| [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large) | 英文   | 1.1 GiB |

> [!NOTE]
>
> 1. [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large) 存在一定的幻觉问题，即容易生成与图片中内容无关的内容。

### 视频字幕模型

| 模型名称                                                     | 支持语言 | 流式推理 | 显存占用 |
| ------------------------------------------------------------ | -------- | -------- | ---- |
| [iic/multi-modal_hitea_video-captioning_base_en](https://www.modelscope.cn/models/iic/multi-modal_hitea_video-captioning_base_en) | 英       |    ❌️      | 尚未测试 |

### 光学字符识别模型

识别一张图片，并将其中包含的文字字符提取出。

| 模型名称                                                               | 支持语言   | 显存占用    |
|--------------------------------------------------------------------|--------|---------|
| [paddlepaddle/PaddleOCR](https://gitee.com/paddlepaddle/PaddleOCR) | 中英法德韩日 | 0.2 GiB |

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
