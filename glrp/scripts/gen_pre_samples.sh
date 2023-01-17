#!/bin/sh

# generate an unannotated file from the current directory CEL gene expression files
cel_files=($(find ./ -maxdepth 1 -name "*.CEL"))
> samples.txt
for i in "${cel_files[@]}"; do
  base_name=$(basename "${i}")
  echo "${base_name}" >> samples.txt
done
