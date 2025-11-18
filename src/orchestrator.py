import glob
import itertools
import os
import shutil
import time

import click
import yaml




def check_statuses(path_to_exp):
    hp = load_experiment(path_to_exp)
    num_exp = len(hp)
    running = 0
    correct_run = 0
    failed_run = 0
    failed_dict=[]
    for summary in glob.glob(
        os.path.join(path_to_exp, "runs", "*", "version_*", "summary.yaml")
    ):
        with open(summary, "r") as f:
            summary_file = yaml.safe_load(f)

            if summary_file["status"] == "RUNNING":
                running += 1
            elif summary_file["status"] == "FAILED":
                failed_run += 1
                failure=                    {
                        "path":summary,
                    }
                
                if "exception" in summary_file:
                    failure["exception"] = summary_file["exception"]
                else:
                    failure["exception"] = "Unkown"
                failed_dict.append(failure)
            elif summary_file["status"] == "SUCCESS":
                correct_run += 1
    total =correct_run + running + failed_run
    click.echo(
        click.style(f"Succeed: {correct_run}", fg="green")+ ", " + click.style(f"Running: {running}", fg="cyan") + ", "+click.style(f" Failed: {failed_run}", fg="red")
    )
    click.secho(
        f"Status: {total} / {num_exp} ({total/num_exp *100}%)", bold=True
        )

    click.echo("==== Failed Experiments ====\n")
    for exp in failed_dict:
        click.echo(f"Experiment path: {exp['path']}")
        click.echo(f"Exception: {exp['exception']}\n")


def find_rerun(path_to_exp):
    click.echo("==== Command to Run ====")
    for summary in glob.glob(
        os.path.join(path_to_exp, "runs", "*", "version_*", "summary.yaml")
    ):
        with open(summary, "r") as f:
            summary_file = yaml.safe_load(f)
            if summary_file["status"] == "FAILED":
                click.echo(summary_file["command"])


def investigate_running(path_to_exp):
    click.echo("==== Investigates Running Experiments ====")
    to_rerun = []
    for summary in glob.glob(
        os.path.join(path_to_exp, "runs", "*", "version_*", "summary.yaml")
    ):
        with open(summary, "r") as f:
            summary_file = yaml.safe_load(f)
            if summary_file["status"] == "RUNNING":
                age_sec = time.time() - os.path.getmtime(summary)
                recently_modified = age_sec < 120
                if not recently_modified:
                    click.echo(summary)
                    to_rerun.append(summary["command"])
    return to_rerun


def compress_exp(path_to_exp):
    exp_name = path_to_exp.removesuffix("/").split("/")[-1]

    output = os.path.join(path_to_exp, exp_name)
    click.echo(f"==== Compressing Experiment {exp_name} to {output} ====")

    shutil.make_archive(output, "zip", path_to_exp)


def get_config_key(circuit_type: str,kernel_size:list[int], *, image_size=None,  **kwargs):
    name = f"{circuit_type}"
    for key, item in kwargs.items():
        if key == "experiment_path":
            continue
        
        name += f"+{item}"

    item="x".join(str(i) for i in kernel_size)
    name += f"+{item}"
    return name


def load_experiment(exp_dir):
    config_grid = os.path.join(exp_dir, "config_grid.yaml")
    with open(config_grid, "r") as yaml_conf:
        explore_grid = yaml.safe_load(yaml_conf)
    explore_grid.update({"experiment_path": [exp_dir]})
    keys, values = zip(*explore_grid.items())
    explore_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    return explore_list
