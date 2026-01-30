from simple_slurm import Slurm


def get_gpu_job_slurm(name, num_gpu=1) -> Slurm:
    exclude_flag = ""
    for i in range(7, 13):
        exclude_flag += f"damnii{str(i).zfill(2)},"

    exclude_flag = exclude_flag[:-1]
    job = Slurm(
        job_name=name,
        cpus_per_task=4,
        mem="128G",
        partition="PGR-Standard",
        gres=f"gpu:{num_gpu}",
        time="24:00:00",
        output=f"./logs/{name}.out",
        exclude=exclude_flag,
    )
    job.add_cmd("source  /home/s2893001/.brc")
    job.add_cmd("hostname")
    job.add_cmd('echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"')
    job.add_cmd("nvidia-smi")
    job.add_cmd("cd  /home/s2893001/projects/pconv")
    job.add_cmd("conda activate ./.conda")
    job.add_cmd('echo "ready to start job"')
    return job


def get_cpu_job_slurm(name) -> Slurm:
    job = Slurm(
        job_name=name,
        cpus_per_task=1,
        mem="8G",
        partition="PGR-Standard",
        time="3:00:00",
        output=f"./logs/{name}.out",
    )
    job.add_cmd("source  /home/s2893001/.brc")
    job.add_cmd("hostname")
    job.add_cmd("cd  /home/s2893001/projects/pconv")
    job.add_cmd("conda activate ./.conda")
    return job
