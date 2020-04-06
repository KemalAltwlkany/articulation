#!/bin/bash

cd ../Tests

declare -a names=("OSY")
for seed in 2
do
	for i in "${names[@]}"
	do
		pipenv run python testparser.py --manual F --seed $seed --problem "$i" --stdID 2105 --save T
	done
done