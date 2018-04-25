for i in */*.csv
do
	echo "this file is $i"
	ticker = "${f/*-/}" 
	awk -F"," 'BEGIN { OFS = "," } {$6="ticker"; print}' WIKI-AVVB.csv  > AVVB_new.csv
done
