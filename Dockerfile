FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:10000", "--timeout", "120"]