#!/bin/bash
# Start only fixture daemons in explicitly mounted, disposable state directories.
# The image root stays read-only. No shared home, cache, or source is chowned.
set -euo pipefail

cp /qualification-config/slurm.conf /qualification-config/slurmdbd.conf /etc/slurm/
if [[ -f /qualification-config/gres.conf ]]; then
    cp /qualification-config/gres.conf /etc/slurm/gres.conf
    chown root:root /etc/slurm/gres.conf
fi
chown root:root /etc/slurm/slurm.conf
chmod 0644 /etc/slurm/slurm.conf
chown slurm:slurm /etc/slurm/slurmdbd.conf
chmod 0600 /etc/slurm/slurmdbd.conf
mkdir -p /run/munge /run/mysqld /run/slurm /var/log/munge
mkdir -p /var/spool/slurmctld /var/spool/slurmd
chown munge:munge /run/munge /var/log/munge /var/lib/munge /etc/munge
chown mysql:mysql /run/mysqld /var/lib/mysql
chown slurm:slurm /run/slurm /var/spool/slurmctld
chmod 0700 /etc/munge
head -c 1024 /dev/urandom > /etc/munge/munge.key
chown munge:munge /etc/munge/munge.key
chmod 0400 /etc/munge/munge.key

children=()
close_daemons() {
    local exit_status=$?
    trap - TERM INT EXIT
    if ((exit_status != 0)); then
        for log_file in /var/log/{mariadb-install,mariadb-console,mariadb-readiness,munge-console,slurmdbd-console,slurmctld-console,slurmd-console,slurmdbd,slurmctld,slurmd}.log; do
            if [[ -f "$log_file" ]]; then
                printf 'Startup log: %s\n' "$log_file" >&2
                tail -c 65536 "$log_file" >&2 || true
            fi
        done
    fi
    if ((${#children[@]})); then
        kill -TERM "${children[@]}" 2>/dev/null || true
        wait "${children[@]}" 2>/dev/null || true
    fi
    return "$exit_status"
}
trap close_daemons TERM INT EXIT
runuser -u munge -- /usr/sbin/munged --foreground > /var/log/munge-console.log 2>&1 &
children+=("$!")
/usr/bin/mariadb-install-db --user=mysql --datadir=/var/lib/mysql --skip-test-db > /var/log/mariadb-install.log 2>&1
/usr/sbin/mariadbd --user=mysql --datadir=/var/lib/mysql --socket=/run/mysqld/mysqld.sock --pid-file=/run/mysqld/mysqld.pid --skip-networking --innodb-buffer-pool-size=32M --innodb-log-file-size=16M > /var/log/mariadb-console.log 2>&1 &
children+=("$!")
database=(/usr/bin/mariadb --no-defaults --protocol=socket --user=root --socket=/run/mysqld/mysqld.sock)
for attempt in {1..45}; do
    "${database[@]}" --execute="SELECT 1" > /var/log/mariadb-readiness.log 2>&1 && break
    if ((attempt == 45)); then exit 1; fi
    sleep 1
done
"${database[@]}" <<'SQL'
CREATE DATABASE slurm_acct_db;
CREATE USER 'slurm'@'localhost' IDENTIFIED BY 'qualification-only';
GRANT ALL ON slurm_acct_db.* TO 'slurm'@'localhost';
SQL
/usr/sbin/slurmdbd -D > /var/log/slurmdbd-console.log 2>&1 &
children+=("$!")
for attempt in {1..45}; do
    /usr/bin/sacctmgr show cluster --noheader --parsable2 >/dev/null 2>&1 && break
    if ((attempt == 45)); then exit 1; fi
    sleep 1
done
/usr/bin/sacctmgr -i add cluster qualification
/usr/bin/sacctmgr -i add account qualification cluster=qualification
/usr/bin/sacctmgr -i add user qualification account=qualification
/usr/sbin/slurmctld -D > /var/log/slurmctld-console.log 2>&1 &
children+=("$!")
/usr/sbin/slurmd -D > /var/log/slurmd-console.log 2>&1 &
children+=("$!")
for attempt in {1..45}; do
    if /usr/bin/sinfo --noheader --format='%T' 2>/dev/null | /bin/grep -q '^idle$'; then
        touch /run/artisan-qualification-ready
        wait -n "${children[@]}"
        exit 1
    fi
    if ((attempt == 45)); then exit 1; fi
    sleep 1
done
