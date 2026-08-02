FROM python:3.11-slim
WORKDIR /app
COPY requirements_web.txt .
RUN pip install --no-cache-dir -r requirements_web.txt
COPY . .
ENV PORT=8000
EXPOSE 8000
CMD ["python", "-u", "interface/app.py"]
