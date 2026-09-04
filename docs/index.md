# artisan-submitit

`artisan-submitit` provides optional SLURM execution providers for Artisan.
Artisan owns pipeline orchestration, records, caching, and artifact commits;
this package transports execution units through `sbatch` or `srun`.

- [Queue-based execution](slurm.md) configures `SlurmRunner` for Submitit job
  arrays.
- [Intra-allocation execution](srun.md) configures `SlurmIntraRunner` inside an
  existing allocation.
- [Migration](migration.md) covers moving from Artisan's former built-in SLURM
  runners.
