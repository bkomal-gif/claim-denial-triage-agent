# Container for the claim-denial triage agent API.
# Works locally and on Hugging Face Spaces (Docker SDK), Render, Google Cloud Run, Azure.
FROM python:3.11-slim

# Run as a non-root user (required by Hugging Face Spaces, good practice everywhere)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
# PORT is set by some platforms (Render, Cloud Run); Hugging Face uses 7860
CMD ["sh", "-c", "uvicorn api:app --app-dir src --host 0.0.0.0 --port ${PORT:-7860}"]
