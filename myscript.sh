#!/bin/bash

cd Tests

declare -a names=("BK1" "FON" "SCH1")
for seed in {0..1}
do
	for i in "${names[@]}"
	do
		pipenv run python testparser.py --manual F --seed $seed --problem "$i" --stdID 15 --save T
	done
done
