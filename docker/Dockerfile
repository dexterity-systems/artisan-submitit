FROM nathanhess/slurm:full-v1.2.0

USER root

# Pixi supplies Python and both editable project environments. Keep the image
# on the same version required by the checked-in lock configuration.
RUN curl -fsSL https://pixi.sh/install.sh | \
    PIXI_HOME=/usr/local PIXI_VERSION=0.66.0 bash
RUN pixi config set detached-environments /pixi-env --system

# SLURM accounting is required for realistic job-state integration tests.
RUN apt-get update && \
    apt-get install -y --no-install-recommends mariadb-server slurmdbd && \
    rm -rf /var/lib/apt/lists/*

COPY slurmdbd.conf /etc/slurm/slurmdbd.conf
RUN chown slurm:slurm /etc/slurm/slurmdbd.conf && \
    chmod 600 /etc/slurm/slurmdbd.conf

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace/artisan-submitit

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
