# Astro
This repository is for sharing some of my simple but effective codes used for astronomical analysis. Description of each code can be found inside the corresponding code file. *NOT regularly maintained anymore.*

## extract_spectral_para_v2.py
### SpecPara
Python class for extracting spectral parameters (e.g., velocity coverage, integrated flux) from velocity-flux spectra.

## HIPE_functions.py

Jython functions for the HIPE software. 

### mkLinemask
Create "line masks" to be used in the spectral data reduction of Herschel/HIFI pointed-observations.

### stat
Return the rms, mean, and median values of different segments of the input spectral dataset with "line masks" defined by mkLinemask.

### unionIntervals
Return a list of union interval(s) of the input intervals, e.g., given [[1,2], [4,6], [5,8]] will result in [[1,2], [4,8]].

### findLineRanges
Simple semi-automatic alogrithm for line and line-like feature detection. Return the frequency ranges for making Linemasks.

## hifi.py

Python class for extracting HIFI observational information from a table of specific format (prepared beforehand). The local-standard-of-rest velocity of each object can be determined automatically for any given line rest frequency.

