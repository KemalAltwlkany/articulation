#!/bin/bash

cd ../Tests

declare -a names=("SCH1")
for seed in {0..8}
do
	for i in "${names[@]}"
	do
		pipenv run python testparser.py --manual F --seed $seed --problem "$i" --stdID 2105 --save T
	done
done