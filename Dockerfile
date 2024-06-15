FROM python:3.9-slim

WORKDIR app
COPY requirements.txt ./
RUN pip install --user --trusted-host pypi.python.org -r requirements.txt
COPY . /app/

EXPOSE 8000

CMD ["python3", "worker/main.py"]
