
import sys
import os
import time
import glob
import yaml

if __name__=="__main__":
    path_to_exp = sys.argv[1]
    print(f"Searching files in {path_to_exp}")
    running=0
    correct_run=0
    failed_run=0
    for summary in glob.glob(os.path.join(path_to_exp, "runs", "*", "version_*", "summary.yaml")):
        with open(summary, "r") as f:
            summary_file = yaml.safe_load(f)

            if summary_file["status"]=="RUNNING":
                running+=1
            elif summary_file["status"]=="FAILED":
                failed_run+=1
                print(summary)
                if "exception" in summary_file:
                    print(summary_file["exception"])
            elif summary_file["status"]=="SUCCESS":
                correct_run+=1
    print(f"Succeed: {correct_run}, Running: {running}, Failed: {failed_run}, Total: {correct_run+running+failed_run}")
    