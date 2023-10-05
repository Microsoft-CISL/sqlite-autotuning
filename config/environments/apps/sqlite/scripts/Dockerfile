FROM benchbase.azurecr.io/benchbase-sqlite:latest

# Install time for some basic timing stats.
USER root

RUN apt-get update \
    && apt-get -y install time \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER containeruser
