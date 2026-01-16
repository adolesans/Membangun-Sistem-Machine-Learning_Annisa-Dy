FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install pandas scikit-learn numpy
CMD ["python", "modelling.py"]