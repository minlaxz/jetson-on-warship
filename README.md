### jetson-on-warship
Frigate NVR running on Jetson Nano (JP4) - A working configuration using Deepstask as detector.

#### Change docker's default runtime to nvidia
```
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```
`sudo systemctl restart docker`

#### Add current user to docker group
`sudo usermod -aG docker $USER && newgrp docker`

Test: `docker ps`

#### Download docker-compose
`curl -x "http://192.168.1.125:2080" -fsSL https://github.com/docker/compose/releases/download/v2.24.5/docker-compose-linux-aarch64 -o docker-compose`

_Since in Myanmar, IDK why tf GFW is blocking githubusercontent.com, so I need to use a proxy in `curl` command's `-x` option._

#### Install docker-compose

`DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}`

`mkdir -p $DOCKER_CONFIG/cli-plugins`

`cp docker-compose $DOCKER_CONFIG/cli-plugins/`

`chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose`

Test: `docker compose version`
