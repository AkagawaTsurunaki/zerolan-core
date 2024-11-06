# Zerolan Live Robot Core

Download development version of `zerolan-data` by running: 

```shell
pip install git+https://github.com/AkagawaTsurunaki/zerolan-data.git@dev
```

Paraformer is not supported by CUDA 11.8, for easy to deploy the model, you can see the Dockerfile located in `./asr/paraformer/Dockerfile`.

Build the image and start the container:

```shell
docker build ./asr/paraformer --tag zerolan-core-asr-paraformer
docker run -d --gpus all -it -p 11001:11001 --name your-zerolan-core-asr-paraformer zerolan-core-asr-paraformer
```

> [!NOTE]
> Some mirrors are used in building phase for faster downloading. Remove them if you DO NOT want to use them.

Config your Nginx

The location of the config file depends on how you install Nginx, for me, it is `/etc/nginx/nginx.conf`

```
server {
        listen 11000;
        server_name HOST:PORT;
        client_max_body_size 20m;
        location /asr {
            proxy_pass http://127.0.0.1:11001;
            proxy_set_header Host $host;             
            proxy_set_header X-Real-IP $remote_addr; 
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;  
        }
        location /llm {
            proxy_pass http://127.0.0.1:11002;
            proxy_set_header Host $host;             
            proxy_set_header X-Real-IP $remote_addr; 
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;  
        }
        location /img-cap {
            proxy_pass http://127.0.0.1:11003;
            proxy_set_header Host $host;             
            proxy_set_header X-Real-IP $remote_addr; 
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;  
        }
        location /ocr {
            proxy_pass http://127.0.0.1:11004;
            proxy_set_header Host $host;             
            proxy_set_header X-Real-IP $remote_addr; 
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;  
        }
        location /vid-cap {
            proxy_pass http://127.0.0.1:11005;
            proxy_set_header Host $host;             
            proxy_set_header X-Real-IP $remote_addr; 
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;  
        }
        location /tts {
            proxy_pass http://127.0.0.1:11006;
            proxy_set_header Host $host;             
            proxy_set_header X-Real-IP $remote_addr; 
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;  
        }
    }
```

Reload your Nginx

```shell
nginx -s reload
```