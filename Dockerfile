FROM python:3.11-slim

COPY ./app /app
COPY ./requirements.txt /tmp/requirements.txt

RUN apt-get clean \
    && apt-get -y update

RUN apt-get -y install nginx \
    && apt-get -y install python3-dev \
    && apt-get -y install build-essential

RUN pip install -r /tmp/requirements.txt --src /usr/local/src

COPY nginx.conf /etc/nginx
COPY start.sh /app/start.sh

WORKDIR /app

CMD ["./start.sh"]