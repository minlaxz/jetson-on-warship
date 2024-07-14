### jetson-on-warship
Frigate NVR running on Jetson Nano (JP4) - A working configuration using Deepstask as detector.


Change docker's default runtime to nvidia
```
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

`sudo service docker stop`
`sudo service docker start`

> Note: We cannot use `restart` command.
