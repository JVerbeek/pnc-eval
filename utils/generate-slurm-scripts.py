from jinja2 import Environment, FileSystemLoader
import os
import glob
import csv

env = Environment(loader=FileSystemLoader('.'))
template = env.get_template('utils/slurmscript.jinja')

for config in glob.glob("experiments/**/**.txt"):
    with open(config, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        indices = [row[0] for row in reader]
    path = "/".join(config.split("/")[:-1])
    params = {
        "config": config, 
        "n_cpu": "4", 
        "time": "10:00:00", 
        "arr_lo": str(min(indices)),
        "arr_hi": str(max(indices))
    }

    script = template.render(**params)

    with open(f"{path}/submit_job.sh", "w") as f:
        f.write(script)

print(f"SLURM script generated: {path}/submit_job.sh")