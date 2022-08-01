FROM python:latest
RUN pip install -U scikit-learn numpy
COPY ./training_data.py /mlop2/
WORKDIR /mlop2/
ENTRYPOINT ["python","trainingdata.py"]