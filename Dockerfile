FROM python:3.10-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./src /code/src
COPY ./models /code/models
COPY ./data /code/data

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

EXPOSE 7860

CMD ["streamlit", "run", "src/frontend/app.py", "--server.address", "0.0.0.0", "--server.port", "7860"]