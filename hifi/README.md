## `hipe_functions.py`

Jython functions for the HIPE software. 

- `mkLinemask`

  Create "line masks" to be used in the spectral data reduction of Herschel/HIFI 
  pointed-observations.

- `stat`

  Return the rms, mean, and median values of different segments of the input 
  spectral dataset with "line masks" defined by mkLinemask.

- `unionIntervals`

   Return a list of union interval(s) of the input intervals, 
   e.g., given `[[1,2], [4,6], [5,8]]` will result in `[[1,2], [4,8]]`.

- `findLineRanges`

   Simple semi-automatic alogrithm for line and line-like feature detection. 
   Return the frequency ranges for making Linemasks.

## `hifi.py`

Python class for extracting HIFI observational information from a table of 
specific format (prepared beforehand). The local-standard-of-rest velocity of 
each object can be determined automatically for any given line rest frequency.

## `plot_spectra_*.py`

Several scripts for plotting spectra and create PDFs.
