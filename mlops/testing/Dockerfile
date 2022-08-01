FROM python:latest
RUN pip install -U scikit-learn
COPY ./testing_data.py /mlop3/
WORKDIR /mlop3/
ENTRYPOINT ["python","testing_data.py"]