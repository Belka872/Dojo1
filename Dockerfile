FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace

COPY requirements-inference.txt ./requirements-inference.txt
RUN pip install --no-cache-dir -r requirements-inference.txt

COPY solution.py ./solution.py
COPY weights ./weights

CMD ["python", "solution.py"]
