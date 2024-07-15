### jetson-on-warship
Running [Frigate](https://github.com/blakeblackshear/frigate) NVR on Jetson Nano (JP4) - A working configuration using Deepstask as a detector.

#### 0.Update and Upgrade
```sh
sudo apt-get update && sudo apt-get upgrade -y
```

#### 1.Change docker's default runtime to `nvidia`
`sudo vim /etc/docker/daemon.json`
```
{
    ...
    "default-runtime": "nvidia" <<< Add this line
}
```
`sudo systemctl restart docker`

#### 2.Add current user to docker group
```sh
sudo usermod -aG docker $USER && newgrp docker
```

Test: `docker ps`

#### 3.Download docker-compose
```sh
curl -x "http://192.168.1.125:2080" -fsSL https://github.com/docker/compose/releases/download/v2.24.5/docker-compose-linux-aarch64 -o docker-compose
```

_In Myanmar, IDK why tf GFW is blocking githubusercontent.com, I need to use a proxy in the `curl` command's `-x` option to reach to._

#### 4.Install docker-compose

```sh
DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
mkdir -p $DOCKER_CONFIG/cli-plugins
cp docker-compose $DOCKER_CONFIG/cli-plugins/
chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose
```
Test: `docker compose version`
