import json
import logging
import os
import shlex

import click
from rich.logging import RichHandler


def rich_logger(level="INFO"):
    """Prepare python logging for rich print."""
    logging.basicConfig(
        level=level, handlers=[RichHandler()], format="%(message)s", datefmt="[%X]"
    )


@click.group
def cli():
    pass


@cli.command()
@click.option(
    "-S",
    "--slurm",
    type=bool,
    is_flag=True,
)
@click.option("-e", "--experiment", type=str, required=True)
@click.option("-g", "--gpus", type=int, default=None)
def start_experiment(slurm, experiment, gpus):
    from src.orchestrator import get_config_key, load_experiment

    explore_list = load_experiment(experiment)
    if gpus is not None:
        import logging

        import ray

        from src.benchmark_logic import benchmark_dataset

        logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(
            logging.FATAL
        )

        os.environ["IN_RAY_TASK"] = "1"

        ray.init(num_gpus=gpus)
        benchmark_dataset_remote = ray.remote(num_gpus=1)(benchmark_dataset)
        futures = [benchmark_dataset_remote.remote(config) for config in explore_list]
        results = ray.get(futures)

        for result in results:
            config = result["config"]
            metrics = result["test_metrics"]
            click.echo(f"✅ Config: {config}")
            for key, value in metrics.items():
                click.echo(f"   {key}: {value:.4f}")
            click.echo("-" * 40)
    else:
        for config in explore_list:
            config.update({"experiment_path": experiment})
            if slurm:
                from src.slurm import get_gpu_job_slurm

                exp_name = experiment.removesuffix("/").split("/")[-1]
                job = get_gpu_job_slurm(exp_name + "___" + get_config_key(**config))
                config_json = json.dumps(config)
                safe_config = shlex.quote(config_json)
                job.add_cmd(f"srun python cli.py run-exp --config {safe_config}")
                job.sbatch()
            else:
                from src.benchmark_logic import benchmark_dataset

                benchmark_dataset(config)


@cli.command()
@click.option(
    "-c",
    "--config",
    type=str,
    required=True,
)
def run_exp(config: str):
    # print("running experiment", flush=True)
    from src.benchmark_logic import benchmark_dataset

    config = json.loads(config)
    benchmark_dataset(config)
    print("=== Task ended successfuly ===")


@cli.command()
@click.option("-e", "--experiment", type=str, required=True)
@click.option("-r", "--run_failed", is_flag=True)
def check_exp(experiment, run_failed):
    from src.orchestrator import check_statuses, find_rerun

    check_statuses(experiment)
    if run_failed:
        to_rerun = find_rerun(experiment)
        # if click.confirm('Do you want to run them as Slurm?'):
        #     for cmd in to_rerun:
        #         job = get_gpu_job_slurm(get_config_key(**config))
        #         job.add_cmd(cmd)
        #         job.sbatch()


@cli.command()
@click.option("-e", "--experiment", type=str, required=True)
def check_running(experiment):
    from src.orchestrator import investigate_running

    investigate_running(experiment)


@cli.command()
@click.option(
    "-S",
    "--slurm",
    type=bool,
    is_flag=True,
)
@click.option("-e", "--experiment", type=str, required=True)
def compress(slurm, experiment):
    from src.orchestrator import compress_exp
    from src.slurm import get_cpu_job_slurm

    click.echo(experiment)
    if slurm:
        exp_name = experiment.removesuffix("/").split("/")[-1]
        job = get_cpu_job_slurm(f"compress-{exp_name}")
        job.add_cmd(f"python cli.py compress --experiment {experiment}")
        job.sbatch()
    else:
        compress_exp(experiment)


@cli.group()
def tune():
    pass


@tune.command()
def mnist():
    import ray

    from src.tune import tune_grid
    from src.tune_grid import mnist_sampling

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    ray.init(num_gpus=2)

    tune_grid(mnist_sampling)


@tune.command()
def cifar():
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

    import ray

    from src.tune import tune_grid
    from src.tune_grid import cifar_config_define

    ray.init(num_gpus=3)

    tune_grid("CIFAR_patch_search", cifar_config_define)


if __name__ == "__main__":
    rich_logger()
    cli()
