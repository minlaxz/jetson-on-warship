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