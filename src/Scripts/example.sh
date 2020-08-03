#!/bin/bash

cd ../Tests

declare -a names=("BK1")
for seed in {0..0}
do
	for i in "${names[@]}"
	do
		for std_id in {0..2}
		do
			echo $seed
			echo $i
			echo $std_id
		done
	done
done
