#!/bin/bash

# Associative array for paired data
declare -A paired_io

# Add paired data to the array
# paired_io["/media/Data3/Jacky/Data/myocardial_cells/20240109 segmentation Jacky/no20_FR_lv7_cTnI-594_CD31-750_PDGFRa-555_Ki67-647_WGA-488.czi"]="/media/Data3/Jacky/Data/myocardial_cells/20240109 segmentation Jacky/no20_FR_lv7_cTnI-594_CD31-750_PDGFRa-555_Ki67-647_WGA-488_res"
paired_io["/media/Data3/Jacky/Data/myocardial_cells/20240109 segmentation Jacky/no17_NS_lv7_cTnI-594_CD31-750_PDGFRa-555_Ki67-647_WGA-488.czi"]="/media/Data3/Jacky/Data/myocardial_cells/20240109 segmentation Jacky/no17_NS_lv7_cTnI-594_CD31-750_PDGFRa-555_Ki67-647_WGA-488_res"

echo "Performing batch colour deconvolution"

# Loop through the paired data
i=0
# Get the length of the array
length=${#paired_io[@]}

for input in "${!paired_io[@]}"; do
    ouptut=${paired_io[$input]}
    echo  "Processing item $((i+1))/$length: Input File: $input, Output Dir: $ouptut"
    python seg.py -i "$input" -o "$ouptut"
    i=$((i+1))
done