#!/bin/bash

cd ../Tests

declare -a names=("TNK")
for seed in 10
do
	for i in "${names[@]}"
	do
		for std_id in {0..2}
		do
			pipenv run python testparser.py --manual T --seed $seed --problem "$i" --stdID $std_id --save T
		done
	done
done