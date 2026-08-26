#!/bin/bash

for dir in $(ls seqbench/experiments); do
	touch seqbench/experiments/$dir/run_experiment.sh
done
