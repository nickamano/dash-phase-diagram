FROM tiangolo/uwsgi-nginx:python3.11

LABEL maintainer= "Nick Amano <namano@umich.edu>"

COPY ./requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

RUN pip uninstall --yes werkzeug
RUN pip install -v https://github.com/pallets/werkzeug/archive/refs/tags/2.0.3.tar.gz

COPY ./app /app

ENV NGINX_WORKER_PROCESSES auto