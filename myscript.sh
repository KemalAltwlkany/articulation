#!/bin/bash

cd Tests

declare -a names=("BK1")
for seed in {0..5}
do
	for i in "${names[@]}"
	do
		pipenv run python testparser.py --manual F --seed $seed --problem "$i" --stdID 15 --save T
	done
done
