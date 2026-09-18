FROM python:3.11-slim
WORKDIR /work
COPY ../requirements.txt /work/requirements.txt
RUN pip install --no-cache-dir -r /work/requirements.txt
COPY . /work
EXPOSE 7860
CMD ["uvicorn", "worker_infer_enhanced:app", "--host", "0.0.0.0", "--port", "7860"]
