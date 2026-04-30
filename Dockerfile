FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser
EXPOSE 7860
ENV FLASK_APP=noshow_iq.api
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=7860"]
