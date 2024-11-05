# Zerolan Live Robot Core

docker build ./asr/paraformer --tag zerolan-core-asr-paraformer

docker run --gpus all -it -p 11001:11001 --name test-asr zerolan-core-asr-paraformer