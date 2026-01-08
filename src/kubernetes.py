from src.orchestrator import load_experiment
import copy
import json
import shlex


def to_json(conf, original_conf):
    js = "{"
    for key, value in conf.items():
        js += f'\\"{key}\\": '
        if isinstance(original_conf[key][0], str):
            js += f'\\"{value}\\",'
        else:
            js += f"{value},"
    js = f"{js[:-1]}}}"
    return js


def create_bash_script(experiment, destination):
    config = load_experiment(experiment)
    script = ""
    for part in experiment.split("/")[::-1]:
        if part != "":
            exp_name = part
            break
    script += f'job_yaml=$(TOTAL_EXP={len(config)} EXP="{exp_name}" envsubst <train-template.yaml)\n'
    script += "job=$(echo \"$job_yaml\" | kubectl create -f - | awk '{print $1}')\n"
    script += f'echo "Launched job for experience {exp_name}"\n'

    with open(destination, "w") as script_file:
        script_file.write(script)
