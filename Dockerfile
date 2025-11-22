FROM python:3.8.10

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE $PORT

CMD gunicorn --workers=2 --bind 0.0.0.0:$PORT app:app
