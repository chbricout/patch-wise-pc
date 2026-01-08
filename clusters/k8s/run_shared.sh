job_yaml=$(TOTAL_EXP=64 EXP="bench-shared-64-restricted" envsubst <train-template.yaml)
job=$(echo "$job_yaml" | kubectl create -f - | awk '{print $1}')
echo "Launched job for experience bench-shared-64-restricted"
