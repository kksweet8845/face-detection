#!/bin/sh



if [ -z "$1" && -z "$2" ]; then
    echo "Usage: source ./gen_csv.sh <the directory path> <output file>"
    return
fi

dir_path="$(cd "$(dirname "$1")"; pwd -P)/$(basename "$1")"
outfile=$2

echo "Generate the csv file into current directory"
find $dir_path/* -not -name Readme.txt | sort | sed -E "s#([0-9]+)_(.*)#\1_\2;\1#g" > $outfile