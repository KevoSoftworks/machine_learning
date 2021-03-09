FROM python:3.8-alpine

WORKDIR /srv/emu

COPY . .

#ENTRYPOINT ["ping", "mail.kevinvonk.com"]

RUN apk update && apk upgrade

RUN pip3 install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "main.py"]
