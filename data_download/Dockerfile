FROM python:3.9-slim-buster
ARG WORK_DIR=/usr/src/app

WORKDIR ${WORK_DIR}
RUN apt-get update && apt-get install -y build-essential gcc pipenv

COPY . ${WORK_DIR}
RUN pip install -r requirements.txt

CMD ["python", "app.py"]
