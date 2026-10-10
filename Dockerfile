FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app1.py .
COPY income_model.pkl .

EXPOSE 8501

CMD ["streamlit", "run", "app1.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
