.PHONY: config clean check_clean


check_clean:
    @echo -n "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]
	
clean_experiments: check_clean
	rm -r experiments

configs: clean_experiments
	./utils/generate-config-txt.sh

slurm_scripts: configs
	./utils/generate-slurm-scripts.py

jobs: slurm_scripts
	