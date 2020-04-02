#!/bin/bash

cd ../Tests

declare -a names=("TNK")
for seed in 100 120 140 160 180 200
do
	for i in "${names[@]}"
	do
		pipenv run python testparser.py --manual F --seed $seed --problem "$i" --stdID 2105 --save T
	done
done