FROM python:3.12-slim

WORKDIR /app

# Install CPU-only PyTorch first (much smaller than full CUDA torch)
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./memcore/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
# Cache HuggingFace models in a volume-mountable location
ENV HF_HOME=/app/.cache/huggingface

EXPOSE 8020

CMD ["python", "-m", "memcore"]
