#!/bin/bash

folder=voxceleb/dev

for file in $(find "$folder" -type f -iname "*.m4a")
do
    name=$(basename "$file" .m4a)
    dir=$(dirname "$file")
    echo ffmpeg -loglevel panic -y -i "$file" "$dir"/"$name".wav
    ffmpeg -loglevel panic -y -i $file $dir/$name.wav
done