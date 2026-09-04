#!/bin/bash
set -e

# Match the development user to the bind-mount owner without changing host
# repository ownership. The defaults match the upstream image's docker user.
ARTISAN_DOCKER_UID=${ARTISAN_DOCKER_UID:-1000}
ARTISAN_DOCKER_GID=${ARTISAN_DOCKER_GID:-1000}
if [ "$(id -g docker)" != "$ARTISAN_DOCKER_GID" ]; then
    groupmod -g "$ARTISAN_DOCKER_GID" docker
fi
if [ "$(id -u docker)" != "$ARTISAN_DOCKER_UID" ]; then
    usermod -u "$ARTISAN_DOCKER_UID" docker
fi

# Configure the single node exposed by the upstream development image.
sed -i "s/<<HOSTNAME>>/$(hostname)/" /etc/slurm/slurm.conf
sed -i "s/<<CPU>>/$(nproc)/" /etc/slurm/slurm.conf
REAL_MEM=$(slurmd -C | grep -oP 'RealMemory=\K[0-9]+')
sed -i "s/<<MEMORY>>/$REAL_MEM/" /etc/slurm/slurm.conf
sed -i \
    's|AccountingStorageType=accounting_storage/none|AccountingStorageType=accounting_storage/slurmdbd|' \
    /etc/slurm/slurm.conf
sed -i '/AccountingStorageType/a AccountingStorageHost=localhost' /etc/slurm/slurm.conf

service munge start

mysqld_safe --skip-networking &
for attempt in $(seq 1 30); do
    mysqladmin ping --silent 2>/dev/null && break
    if [ "$attempt" -eq 30 ]; then
        echo "ERROR: MariaDB did not start." >&2
        exit 1
    fi
    sleep 1
done

mysql -e "CREATE DATABASE IF NOT EXISTS slurm_acct_db;"
mysql -e "CREATE USER IF NOT EXISTS 'slurm'@'localhost' IDENTIFIED BY 'slurm';"
mysql -e "GRANT ALL ON slurm_acct_db.* TO 'slurm'@'localhost';"

slurmdbd
for attempt in $(seq 1 30); do
    sacctmgr show cluster --noheader --parsable2 2>/dev/null && break
    if [ "$attempt" -eq 30 ]; then
        echo "ERROR: slurmdbd did not start." >&2
        exit 1
    fi
    sleep 1
done

sacctmgr -i add cluster cluster 2>/dev/null || true
sacctmgr -i add account default cluster=cluster 2>/dev/null || true
sacctmgr -i add user docker account=default 2>/dev/null || true

service slurmd start
service slurmctld start

for attempt in $(seq 1 30); do
    if squeue --noheader 2>/dev/null; then
        echo "SLURM is ready."
        break
    fi
    if [ "$attempt" -eq 30 ]; then
        echo "ERROR: SLURM did not become ready in 30s." >&2
        exit 1
    fi
    sleep 1
done

chown -R docker:docker /home/docker /pixi-env 2>/dev/null || true

exec runuser -u docker -- "$@"
