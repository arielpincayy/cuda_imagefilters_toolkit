#!/bin/bash

DATA_DIR="data"
RESULTS="results.txt"
mkdir -p "$DATA_DIR/gray" "$DATA_DIR/sobel" "$DATA_DIR/gaussian"
touch $RESULTS
> $RESULTS

for file in data/originals/*; do
    filename=$(basename "$file")
    
    gray_path="$DATA_DIR/gray/$filename"
    sobel_path="$DATA_DIR/sobel/$filename"
    gaussian_path="$DATA_DIR/gaussian/$filename"
    original_path="$file"

    ./output/main "$file" "$gray_path" "$sobel_path" "$gaussian_path" >> $RESULTS
    echo "------------------------------------------------------" >> $RESULTS

    echo "Procesado: $filename"
done
