FROM python:3

LABEL maintainer="Pranay Payal"

WORKDIR /app

COPY './requirements.txt' .
COPY './trained-model.pt' ..

# RUN apt-get install libgtk2.0-dev pkg-config -yqq 

RUN pip install --no-cache-dir --upgrade pip
RUN apt-get update && apt-get install -y cmake
RUN pip install --no-cache-dir dlib
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]