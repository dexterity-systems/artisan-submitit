# Tutorials

Run Artisan's built-in operations on SLURM, then inspect the data they produce.
These notebooks import `artisan` and `artisan_submitit` from your Python
environment. Each notebook runs independently and includes assertions that check
its results.

| Tutorial | What you will run | Where the notebook kernel runs |
| --- | --- | --- |
| [Your first SLURM pipeline](01-first-slurm-pipeline.ipynb) | Generate CSV files, transform them, and calculate metrics with `SlurmRunner`. | A login or workflow node with SLURM command access. |
| [Inside an allocation](02-inside-an-allocation.ipynb) | Run the same workflow with `SlurmIntraRunner`. | Inside an allocation created by `salloc` or `sbatch`. |
| [Batching and managed commands](03-batching-and-commands.ipynb) | Pack CSV inputs into workers and run `CsvHead` as a managed command. | A login or workflow node with SLURM command access. |

## Prepare your environment

You need Python 3.12 or newer, both packages installed in an environment visible
at the same path on the manager and compute nodes, and a shared filesystem for
the tutorial's data and logs. Queue-based tutorials also need working SLURM
accounting through `sacct`. The examples use CPU resources only.

With released packages, install the runtime and notebook tools into your Python
environment:

```bash
python -m pip install artisan-submitit jupyterlab
```

Open the notebooks from a checkout or download of this repository. From its root,
set a shared output location and your cluster's partition:

```bash
export ARTISAN_TUTORIAL_ROOT=/shared/your-user/artisan-submitit-tutorials
export ARTISAN_SLURM_PARTITION=cpu
# Set this if your cluster requires an account.
export ARTISAN_SLURM_ACCOUNT=your-account
python -m jupyterlab docs/tutorials
```

Replace the example path, partition, and account with your site's values. Select
the Python kernel from the environment where you installed the packages. Each
notebook prints the imported package versions and paths before submitting work.

For development, the repository's Pixi environment includes both editable
packages and Jupyter. Follow the [development checkout setup](https://github.com/dexterity-systems/artisan-submitit/blob/main/README.md#development-and-docker),
then run:

```bash
~/.pixi/bin/pixi install --locked -e tutorials
~/.pixi/bin/pixi run --locked -e tutorials tutorials
```

The same environment variables configure both installation methods. If
`ARTISAN_TUTORIAL_ROOT` is unset, notebooks use `tutorial-runs` beneath their
working directory. Use that default only when the directory is shared with
compute nodes.

## Start the allocation tutorial

Start Jupyter from the shell that owns the allocation. For example, with the
environment variables above and any account options required by your site:

```bash
# Add --account=your-account if your site requires it.
salloc --partition="$ARTISAN_SLURM_PARTITION" --ntasks=2 --mem=4G --time=00:15:00
~/.pixi/bin/pixi run --locked -e tutorials tutorials
```

For the pip installation, use `python -m jupyterlab docs/tutorials` in that shell.
Follow your site's instructions for connecting to Jupyter. Opening `salloc` in
another terminal does not put an existing notebook kernel inside the allocation.
Close Jupyter and exit the allocation shell when you finish.

## Rerun a notebook

Each notebook uses its own directory under the configured tutorial root. Its
setup cell clears that notebook's previous results so assertions also hold on a
second run. Finish or cancel previous work before rerunning setup. Keep any
results you want to retain elsewhere, then restart the kernel and run all cells
in order.

## Validate tutorial changes

The tutorial tests execute the notebooks themselves and retain executed copies
under `slurm_logs/tutorials/`. From a shell with scheduler access, run:

```bash
~/.pixi/bin/pixi run --locked -e tutorials test-tutorial-sbatch
~/.pixi/bin/pixi run --locked -e tutorials test-tutorial-srun
```

The second task creates an allocation for the kernel. Set
`ARTISAN_TUTORIAL_ROOT` to a shared location for tests as well as interactive use.
For the local test cluster, follow the [Docker instructions](https://github.com/dexterity-systems/artisan-submitit/blob/main/docker/README.md).
Commit notebooks with empty outputs. The separate
`~/.pixi/bin/pixi run --locked -e docs docs-build` task renders the site; notebook
execution is checked by the scheduler tests.
