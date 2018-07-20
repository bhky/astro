"""
HIPE functions.
"""

def mkLinemask(obs, freqRanges=None, lineFreqs=None, vexp=15, usevexp=True, \
               lineWidths=None, vlsr=0):
   """
   mkLinemask(obs, freqRanges=None, lineFreqs=None, vexp=15, usevexp=True, \
              lineWidths=None, vlsr=0)

   Make Linemask table for HIFI pointed-observation spectra by specifying the
   frequency ranges for masking. Alternatively, users can provide the line 
   frequencies and the evenlope expansion velocity, or the corresponding 
   line widths for calculating the frequency ranges.

   If the Linemask created by this function is to used with fitBaseline(),
   remember to set domask=0 and doreuse=False for the latter, i.e.,

   fitBaseline(..., domask=0, doreuse=False, maskTable=Linemask). 

   It is alright to set domask=2 if one would like HIPE to determine 
   automatically whether additional masks are needed. However, it is found 
   that domask=1 may nullify some of the masks in Linemask. Reason unknown.


   PARAMETERS

   obs: HIPE ObservationContext

   freqRanges: list
      List of tuples (or lists) containing the frequency ranges for masking,
      i.e., [(freq_start1, freq_end1), (freq_start2, freq_end2), ... ].
      If this parameter is given, the other parameters below are not used.
      The default value is None.

   lineFreqs: list, array, or sequence-like of double
      The frequencies (GHz) of all the lines. The default value is None.

   vexp: double
      The envelope expansion velocity (km/s) of the object. If usevexp=True,
      this velocity will be used to calculate the line widths, and lineWidths
      will be neglected. If usevexp=False, then this parameter is neglected.
      The default value is 15 (km/s).

   usevexp: boolean
      If usevexp=True (default), then the line widths of the given lines in
      lineFreqs will be caculated with vexp.

   lineWidths: list or sequence-like of double
      The full-width-at-zero-intensity (GHz) of all the lines can be provided
      manually with a list in 1-to-1 correspondance to lineFreqs. Note that 
      this parameter is neglected if usevexp=True (default).

   vlsr: double
      The local-standard-of-rest velocity (km/s) of the object. The default 
      value is 0 (km/s).


   RETURN
   
   Tuple (Linemask, freq_pairs)

   Linemask: TableDataset
      The Linemask table

   freq_pairs: Double1d array
      Array containing pairs of frequencies defining the mask ranges.
      If freqRanges is used, then this is equivalent to the output of
      numpy.ravel(freqRanges) in Python.

   """

   ## Create an empty table
   Linemask = TableDataset(description="Line masks created by mkLinemask().")
   
   ## Create all the empty columns in the table
   col_list = ["freq_1", "freq_2", "weight", "origin", "peak", "median", \
                   "dataset", "scan"]
   for col_name in col_list:
      Linemask[col_name] = Column(Double1d())

   ## Create meta data
   Linemask.meta["HifiTimelineProduct"] = StringParameter()
   Linemask.meta["dataset"] = DoubleParameter()
   Linemask.meta["scan"] = DoubleParameter()
   Linemask.meta["subband"] = DoubleParameter()

   ## Define an array that will carry the freq1, freq2 pairs
   freq_pairs = Double1d()

   ## Create the mask parameters
   if freqRanges is not None:
      l = freqRanges
   else:
      l = lineFreqs
   for i in range(len(l)):
      ## If freqRanges is given, use it
      if freqRanges is not None:
         freq1a = i[0]
         freq2a = i[1]
      ## Else, use lineFreqs
      else:
         ## Adjust doppler shift for each frequency
         c = 299792.458
         freq_vlsr = (1 - vlsr / c) * lineFreqs[i]
         ## Calculate the starting and ending frequencies of each mask
         if usevexp:
            ## Use vexp to calculate line widths
            freq_1a = (1 - (vlsr + vexp) / c) * lineFreqs[i]
            freq_2a = (1 - (vlsr - vexp) / c) * lineFreqs[i]
         else:
            ## Use the lineWidths list 
            freq_1a = freq_vlsr - lineWidths[i] / 2
            freq_2a = freq_vlsr + lineWidths[i] / 2
      ## Create another set of frequenices for the other side band
      loFreq = obs.meta["loFreqAvg"].double
      if freq_1a != loFreq:
         freq_1b = 2 * loFreq - freq_1a
      if freq_2a != loFreq:
         freq_2b = 2 * loFreq - freq_2a
      ## Append freq_pairs list
      for freq in [freq_1a, freq_2a, freq_2b, freq_1b]:
         freq_pairs.append(freq)
      ## Add rows of mask parameters to the table
      Linemask.addRow([freq_1a, freq_2a, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0])
      Linemask.addRow([freq_2b, freq_1b, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0])

   ## Return a tuble containing the Linemask table and freq_pairs array
   return (Linemask, freq_pairs)


def stat(sp, excludePairs=None, **kwargs):
   """
   rms(sp, excludePairs=None, **kwargs)

   Return three arrays in a list containing the rms, mean, and median values 
   of different segments of the given spectral dataset. The values are 
   computed by the build-in statistics() function. The definition of segments 
   can be changed depending on the value of the "mode" parameter of 
   statistics().


   PARAMETERS

   sp: HIPE SpectrumContainer

   excludePairs: list, array, or sequence-like of Double
      Pairs of starting and ending sky frequencies defining the range to be
      excluded from rms computation. The freq_pairs output from mkLinemask()
      can be put here. The format is [freq1a, freq1b, freq2a, freq2b, ...].
      Either this parameter or "exclude" in statistics() should be used.
      The default value is None.

   **kwargs: dict
      Additional keyword arguments are passsed to statistics().


   RETURN

   list [rms_array, mean_array, median_array]

   rms_array: Double1d array
      Array containing the rms values of the segments.
 
   mean_array: Double1d array
      Array containing the mean values of the segments.

   median_array: Double1d array
      Array containing the median values of the segments.

   """

   if excludePairs is not None:
      ## Change sky_freq_pairs into list-of-tuples form
      freq_tuples_list = []
      for i in range(0, len(excludePairs), 2):
         freq_tuples_list.append((excludePairs[i], excludePairs[i+1]))
      ## Compute spectrum stat with the "exclude" parameter
      stats = statistics(ds=sp, exclude=freq_tuples_list, **kwargs)
   else:
      ## Compute spectrum stat
      stats = statistics(ds=sp, **kwargs)

   ## Store the stat values in a list
   rms_list = \
   [stats[col].data[0] for col in stats.columnNames if 'rms_' in col]
   mean_list = \
   [stats[col].data[0] for col in stats.columnNames if 'mean_' in col]
   median_list = \
   [stats[col].data[0] for col in stats.columnNames if 'median_' in col]

   ## Return a list of arrays containing the stat values
   rms_array = Double1d(rms_list)
   mean_array = Double1d(mean_list)
   median_array = Double1d(median_list)
   return [rms_array, mean_array, median_array]


def unionIntervals(intervals):
   """
   unionIntervals(intervals)

   Return a list of union interval(s) of the input intervals, e.g., 
   given [[1,2], [4,6], [5,8]] will result in [[1,2], [4,8]].


   PARAMETERS

   intervals: list or sequence-like
      list of lists/tuples defining the intervals, e.g., [[0,1], [5,8], ...]

   RETURN

   union_intervals: list
      list of list(s) defining the union interval(s)
   
   """
   union_intervals = []
   for interval in sorted(intervals):
      interval = list(interval)
      if union_intervals and union_intervals[-1][1] >= interval[0]:
         union_intervals[-1][1] = max(union_intervals[-1][1], interval[1])
      else:
         union_intervals.append(interval)
   return union_intervals


def findMaskRanges(obs, backend, channelNum=100, sigmaNum=1.5, excludeNum=5, \
                   widthFactor=2):
   """
   findMaskRanges(obs, backend, channelNum=100, sigmaNum=1.5, excludeNum=5, \
                  widthFactor=2)

   Simple semi-automatic alogrithm for line and line-like feature detection. 
   Return the frequency ranges for making Linemasks.


   PARAMETERS

   obs: HIPE ObservationContext

   backend: str
      Must be one of the following: 'WBS-H-LSB', 'WBS-H-USB', 'WBS-V-LSB', 
      'WBS-V-USB', 'HRS-H-LSB', 'HRS-H-USB', 'HRS-V-LSB', 'HRS-V-USB'.

   channelNum: int
      Number of channels to be included in one "channel group" when 
      considering possible detections. The default value is 100.

   sigmaNum: double
      If the (mean) flux of a channel group is sigmaNum larger or smaller 
      than the mean flux of all the channel groups, this group is labelled as 
      line containing. The default value is 1.5.

   excludeNum: int
      The excludeNum channels with the largest flux and smallest flux will be
      excluded from the calculation of the mean channel group flux. The Default
      value is 5.

   widthFactor: double
      The factor to be multiplied to the "line width" predicted from the
      channel groups. Note that the original predicted value may not reflect
      the real line width, it is better to be more conservative to make it
      wider especially when making Linemasks. The default value is 2.


   RETURN

   freqRanges: list
      List containing tuple(s) of the two frequencies defining a range where 
      a mask should be applied, in the format of 
      [(freq_start1, freq_end1), (freq_start2, freq_end2), ... ].
      This parameter can be directly input to mkLinemask.

   """
   
   ## Read Level 2.5 spectrum from obs
   sp_25 = obs.refs["level2_5"].product.refs["spectrum"].product.\
           refs["spectrum_"+backend].product["dataset"].copy()
   freq_array = sp0_25["wave"].data
   flux_array = sp0_25["flux"].data

   ## Compute the average flux in each divided group of channels
   ## The last group will include also the remainding channels (if any), i.e.,
   ## more channels than the other groups
   flux_mean_array = Double1d()
   n = channelNum
   start_ch_list = range(0, len(flux_array), n)[:-1]
   for i in start_ch_list:
      if i != start_ch_list[-1]:
         if not IS_NAN(MEAN(flux_array[i:i+n])):
            flux_mean_array.append(MEAN(flux_array[i:i+n]))
         else:
            flux_mean_array.append(0)
      else:
         if not IS_NAN(MEAN(flux_array[i:])):
            flux_mean_array.append(MEAN(flux_array[i:]))
         else:
            flux_mean_array.append(0)

   ## Compute mean and std of flux_mean_array after removing excludeNum (int)
   ## max and min values (respectively), then compare all the values of the
   ## orginal flux_mean_array to find the indices of the values deviate from
   ## mean by sigma*sigmaNum or larger
   flux_mean_list = list(flux_mean_array)
   for i in range(excludeNum):
      flux_mean_list.remove(max(flux_mean_list))
      flux_mean_list.remove(min(flux_mean_list))
   mean = MEAN(flux_mean_list)
   sigma = STDDEV(flux_mean_list)
   deviate_index_pair_list = [[list(flux_mean_array).index(f)*n, \
                             list(flux_mean_array).index(f)*n + n] \
                             for f in flux_mean_array \
                             if f > (mean + sigma*sigmaNum) \
                             or f < (mean - sigma*sigmaNum)] 

   ## Get the corresponding frequency ranges after applying the width factor
   freqRanges = []
   for index_pair in deviate_index_pair_list:
      half_width = abs(freq_array[index_pair[1]]-freq_array[index_pair[0]])/2
      cent_freq = (freq_array[index_pair[0]]+freq_array[index_pair[1]])/2
      start_freq = cent_freq - half_width * widthFactor
      end_freq = cent_freq + half_width * widthFactor
      freqRanges.append((start_freq, end_freq))

   ## Find the union ranges
   freqRanges = unionIntervals(freqRanges) 
      
   ## Return freqRanges
   return freqRanges


