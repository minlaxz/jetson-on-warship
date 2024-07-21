FROM ultralytics/ultralytics:latest-jetson-jetpack4

WORKDIR /app
RUN mkdir /app/models

RUN apt-get update \
    && apt-get install -y \
    vim \
    nmap \
    curl \
    && apt-get clean

COPY ./requirements.txt requirements.txt
COPY ./app.py app.py

RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

RUN curl -fsSL https://github.com/ultralytics/yolov5/archive/master.zip -o /root/.cache/torch/hub/master.zip
COPY ./models/yolov5s.pt /app/models/yolov5s.pt

CMD ["python3", "app.py"]
