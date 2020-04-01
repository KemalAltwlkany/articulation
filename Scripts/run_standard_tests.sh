#!/bin/bash

cd ../Tests

declare -a names=("BK1")
for seed in {0..0}
do
	for i in "${names[@]}"
	do
		for std_id in {0..2}
		do
			pipenv run python testparser.py --manual T --seed $seed --problem "$i" --stdID $std_id --save T
		done
	done
done