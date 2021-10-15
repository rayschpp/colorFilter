# syntax=docker/dockerfile:1

FROM python:3.9.7-alpine3.14

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY src/main/python .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]

