# Use Python 3.12 slim image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Set working directory
WORKDIR /app

# Install system dependencies, download and compile TA-Lib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    ca-certificates \
    autotools-dev \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && cp /usr/share/misc/config.guess . \
    && cp /usr/share/misc/config.sub . \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz \
    && apt-get purge -y --auto-remove build-essential wget autotools-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and source code
COPY pyproject.toml README.md /app/
COPY src/ /app/src/

# Install python dependencies
RUN pip install --no-cache-dir -e .

# Create volume mounting point for DuckDB database and reports
RUN mkdir -p /app/data

# Default entry point command
ENTRYPOINT ["python", "-m", "vibe_trading.cli"]
CMD ["live"]
