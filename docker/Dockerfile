FROM continuumio/miniconda3

WORKDIR /app
COPY ./requirements.txt ./
RUN pip install -r requirements.txt
COPY ./data ./data
COPY ./main.py ./

ENTRYPOINT [ "python main.py" ]
