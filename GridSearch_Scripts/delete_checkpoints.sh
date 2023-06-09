#!/bin/bash

# Define the source directory where the directories are located
source_dir="../Data/MIL_v2"

# Loop through the directories starting with 'MIL'
for dir in 'MIL-'*; do
    if [ -d "$dir" ]; then
        # Extract the directory name
        dir_name=$(basename "$dir")
        zip_file="${dir_name}.zip"
        echo "$zip_file"
             
        # Create a zip file with the same name as the directory
        rm -rf "$zip_file" "$dir_name"
    fi
done