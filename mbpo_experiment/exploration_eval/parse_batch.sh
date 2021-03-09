#! /bin/bash
for index in {5..40..5}
	do
		echo checkpoint $index
		#python parse.py -d grid_sac3x3_data -i $index
		#python parse.py -d grid_sac3x3distinct_data -i $index
		#python parse.py -d grid_sac_data -i $index
		python parse.py -d reacher_sac3x3distinct_data -i $index
done


