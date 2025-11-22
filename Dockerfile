FROM python:3.8.10

# Install system deps required by OpenCV (cv2)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE $PORT

CMD gunicorn --workers=3 --bind 0.0.0.0:$PORT app:app
