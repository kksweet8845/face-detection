#!/bin/sh



if [ -z "$1" ]; then
    echo "Usage: source ./gen_csv.sh <the directory path>"
    return
fi

dir_path="$(cd "$(dirname "$1")"; pwd -P)/$(basename "$1")"

echo "Generate the csv file into current directory"
find $dir_path/* -not -name Readme.txt | sort | sed -E "s#([0-9]+)(.*)#\1\2;\1#g" > ./csv.ext