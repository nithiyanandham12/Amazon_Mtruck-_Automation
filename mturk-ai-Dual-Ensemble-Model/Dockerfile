# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# This Dockerfile runs the Gradio app on port 7860

FROM python:3.9

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements first for better caching
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all application files
COPY --chown=user . /app

# The app.py launches Gradio with demo.queue().launch()
# Gradio will automatically listen on port 7860
CMD ["python", "app.py"]


