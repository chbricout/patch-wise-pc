import os
import glob

from src.benchmark_logic import BenchPCImage


def bench_sampling_experiment(experiment_path):
    # load experiment and list runs
    experiments = os.listdir(os.path.join(experiment_path, "runs"))

    # for each run, load model and generate samples (either cpu or gpu)
    for exp in experiments:
        checkpoints = glob.glob(
            os.path.join(experiments, "runs", exp, "version_0", "checkpoints", "*.ckpt")
        )
        if len(checkpoints) == 0:
            print("No checkpoints")
        else:
            module = BenchPCImage.load_from_checkpoint(
                checkpoints[0], map_location="cpu"
            )

    # log runtime, flops, samples and sample quality in a sample_metric.csv file
