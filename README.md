# Zerolan Live Robot Core

Download development version of `zerolan-data` by running: 

```shell
pip install git+https://github.com/AkagawaTsurunaki/zerolan-data.git@dev
```

Paraformer is not supported by CUDA 11.8, for easy to deploy the model, you can see the Dockerfile located in `./asr/paraformer/Dockerfile`.

Build the image and start the container:

```shell
docker build ./asr/paraformer --tag zerolan-core-asr-paraformer
docker run --gpus all -it -p 11001:11001 --name your-zerolan-core-asr-paraformer zerolan-core-asr-paraformer
```

> [!NOTE]
> Some mirrors are used in building phase for faster downloading. Remove them if you DO NOT want to use them.