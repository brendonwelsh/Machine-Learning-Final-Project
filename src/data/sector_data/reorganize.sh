for i in *.csv
do
	echo "this file is $i"
	tick="${i/*-/}"
	ticker="${tick/.*/}"
	echo $ticker
	awk -F"," 'BEGIN { OFS = "," } {$6=ticker; print}' $i > $ticker_new.csv
done
