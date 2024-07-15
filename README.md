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


The problem is I cannot start the container.

#### Digging into Frigate container startup.
First, this [docker/tensorrt/detector/rootfs/etc/s6-overlay/s6-rc.d/trt-model-prepare/run](https://github.com/blakeblackshear/frigate/blob/b7cf5f4105e3b89eaaac5adddf00ade1c704597d/docker/tensorrt/detector/rootfs/etc/s6-overlay/s6-rc.d/trt-model-prepare/run#L80) which will run right after "Downloading yolo weights" prompt.

```
The script "download_yolo.sh" is not in the Frigate repository so it must be from somewhere else.
```

Second, this [docker/tensorrt/detector/tensorrt_libyolo.sh](https://github.com/blakeblackshear/frigate/blob/b7cf5f4105e3b89eaaac5adddf00ade1c704597d/docker/tensorrt/detector/tensorrt_libyolo.sh#L8)
is downloading another repository from [tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos/tree/master)


Third, this [docker/tensorrt/detector/tensorrt_libyolo.sh](https://github.com/blakeblackshear/frigate/blob/b7cf5f4105e3b89eaaac5adddf00ade1c704597d/docker/tensorrt/detector/tensorrt_libyolo.sh#L21)
copying a folder into `SCRIPT_DIR` which is `SCRIPT_DIR="/usr/local/src/tensorrt_demos"`

Now this [docker/tensorrt/detector/rootfs/etc/s6-overlay/s6-rc.d/trt-model-prepare/run](https://github.com/blakeblackshear/frigate/blob/b7cf5f4105e3b89eaaac5adddf00ade1c704597d/docker/tensorrt/detector/rootfs/etc/s6-overlay/s6-rc.d/trt-model-prepare/run#L77-L80) make sense, it's actually calling this [script](https://github.com/jkjung-avt/tensorrt_demos/blob/master/yolo/download_yolo.sh)

And again in the script, it runs `wget` to these `raw.githubusercontent.com` domains which are blocked by GFW.
