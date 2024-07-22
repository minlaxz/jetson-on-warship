FROM ultralytics/ultralytics:latest-jetson-jetpack4

WORKDIR /app
RUN mkdir /app/models
RUN mkdir -p /root/.cache/torch/hub/

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

CMD ["gunicorn", "-w", "4", "--threads", "2", "-b", "0.0.0.0:5000", "app:lightstack"]
