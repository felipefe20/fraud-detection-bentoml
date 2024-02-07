FROM tiangolo/uvicorn-gunicorn-fastapi:latest
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY ./src /app
RUN python /app/train_forecaster.py  # Run the train_forecaster script
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]