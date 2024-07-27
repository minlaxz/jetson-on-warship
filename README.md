Ultralytics YOLOv8 as Frigate Detector in Docker on Jetson Nano
===

[Frigate](https://github.com/blakeblackshear/frigate) NVR (Network Video Recorder) on Jetson Nano (JP4) - using YOLOv8 as a detector.

> Frigate officially supports `Deepstack` as one of their Object detectors but Deepstack isn't actively maintained and here's yet another one, lightstack (only focusing on object detection) and trained on YOLOv8 nano model.

### Setup Docker and Docker Compose

#### 0.Super Duper `update` and `upgrade`
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

#### 2.Add current `user` to `docker group`
```sh
sudo usermod -aG docker $USER && newgrp docker
```

#### 3.Download docker-compose
```sh
curl -x "http://192.168.1.125:2080" -fsSL https://github.com/docker/compose/releases/download/v2.24.5/docker-compose-linux-aarch64 -o docker-compose
```

_In Myanmar, IDK why **Great Firewall** is blocking `githubusercontent.com`, so I need to use a proxy in the `curl` command's `-x` option to bypass GFW._

#### 4.Install docker-compose

```sh
DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
mkdir -p $DOCKER_CONFIG/cli-plugins
cp docker-compose $DOCKER_CONFIG/cli-plugins/
chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose
```
---

### Ultralytics YOLOv8 on Jetson nano using Docker [ref](https://docs.ultralytics.com/guides/nvidia-jetson/#best-practices-when-using-nvidia-jetson)


#### 0.Best practices
```bash
sudo nvpmodel -m 0 # Running in MAX_5A
sudo jetson_clocks
sudo apt update && sudo pip3 install jetson-stats
sudo reboot
jtop
```

#### 1.Ultralytics YOLOv8
_Export .pt to .engine first if you're using with Nvidia devices for better performance._
```python
from ultralytics import YOLO

# Load a YOLOv8n PyTorch model
model = YOLO("yolov8n.pt")

# Export the model
model.export(format="engine")  # creates 'yolov8n.engine'

# Load the exported TensorRT model
trt_model = YOLO("yolov8n.engine")

# Run inference
results = trt_model("https://ultralytics.com/images/bus.jpg")
```

#### 2.Running lightstack-api


```bash
docker run --rm -it --runtime nvidia --network host --ipc host -v $(pwd)/models:/app/models ghcr.io/minlaxz/lightstack-api:yolov8-jp4
```
---

Citation

```BibTeX
@software{yolov8_ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics YOLOv8},
  version = {8.0.0},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}
```