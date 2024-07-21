FROM ultralytics/ultralytics:latest-jetson-jetpack4


# Install dependencies
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install Flask

COPY ./app.py ./app.py

CMD ["python3", "app.py", "--port", "5001"]
