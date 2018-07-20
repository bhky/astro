#!/usr/bin/env python2.7

import numpy as np


class SpecPara(object):
   """
   SpecPara(input_file, rms, n_sigma=3, n_bin=2, sep=' ', skiprows=0)


   -- Description

   This class can be used to create SpecPara objects each contains 
   a two-column (velocity, flux) spectrum read from text file. Common
   spectral parameters (see below) can then be extracted. The code was 
   originally designed for handling maser spectra, but it should work 
   also on other similar spectral data.
   
   One key feature of this code is that it could automatically skip the
   strong Radio Frequency Interference (RFI) signals that cannot be avoided 
   in some wide band observations. A RFI signal is usually very narrow 
   (even mono-frequency) which occupies only a single velocity channel. 
   The code has an algorithm to discard channels with such signals.
   
   
   -- IMPORTANT notes
   
   (1) All entries must be numeric.
   (2) There must be no repeating values in the "velocity" column.
   (3) The velocity entries must go down the file in ascending order.
   (4) It is assumed that there is only ONE maximum flux value in the spectrum.
   (5) If there is no emission, all the spec() method will return an array of
       numpy.nan values.


   -- Parameters

   input_file: file, str
      A two-column (velocity, flux) text file of rms-subtracted spectral data. 
      Velocity is in the unit of km/s; flux in Jy. The file is loaded in by
      numpy.loadtxt(). By default no header is assumed, but it can be changed
      by using the skiprows parameter.

   rms: float
      The known rms of that spectrum.

   n_sigma: int, optional
      Number of sigma. Default is 5.

   n_bin: int, optional
      Mininum number of consecutive velocity bins with flux above n_sigma * rms 
      that will be treated as real emission peak. Default and minimum value 
      is 2.

   sep: str: optional
      Separater/Delimiter of the input file. Default is ' '.


   -- Attributes

   vel: array
      Containing the velocities of the emission channels. Non-emission channels
      are discarded.
   flux: array
      Containing the flux values corresponding to the above velocities.
   bins: array
      Containing the velocity bin number (0 represents the first one of the 
      input file) corresponding to the above velocity and flux values. 
      

   -- Method

   spec()
      Returns: array
      ([velmin, velmax, velcov, velave, pvel, pflux, int_flux, fvel, ncom])

      velmin, velmax: minimum, maxium velocities

      velcov: velocity coverage

      velave: average or "mid-point" velocity

      pvel, pflux: velocity and flux of the highest emission peak

      int_flux, fvel: velocity integrated flux, flux-weighted mean velocity, 

      ncom: number of separate velocity components above n_sigma * rms.

   -----------------------------------
   Updated 2017-10-26 Bosco Yung
   """

   def __init__(self, input_file, rms, n_sigma=3, n_bin=2, sep=' ', \
                skiprows=0):
      ### Check input
      if n_bin < 2:
         print 'n_bin has to be >= 2'
         exit(1)

      ### Read input file
      vel_all, flux_all = np.loadtxt(input_file, delimiter=sep, unpack=True,
                                     usecols=(0,1), skiprows=skiprows)
      ### Discard the channels not meeting up the detection criteria, 
      ### make new lists
      vel = []; flux = []; bins = []
      rms = float(rms)
      for i in range(len(flux_all) - n_bin + 1):
         if all(flux_all[j] >= rms * n_sigma for j in range(i,(i + n_bin))):
            for k in range(i,(i + n_bin)):
               if vel.count(vel_all[k]) == 0:
                  vel.append(vel_all[k])
                  flux.append(flux_all[k])
                  bins.append(k)
      self.vel = np.array(vel)
      self.flux = np.array(flux)
      self.bins = np.array(bins)

   def spec(self):

      vel=self.vel; flux=self.flux; bins=self.bins
      if len(vel) != 0:

         ### min and max velocities
         velmin = min(vel)
         velmax = max(vel)
         ### Velocity coverage
         velcov = velmax - velmin
         ### Average or mid-point velocity coverage
         velave = (velmin + velmax) / 2.0
         
         ### Peak velocity and flux
         ### It is assumed here that there is only ONE max flux value
         pvel = vel[list(flux).index(max(flux))] 
         pflux = max(flux)

         ### Integrated flux, flux-weighted mean velocity, 
         ### and number of velocity components
         ### The "trapeziums" method is used to find iflux
         ### First need to be sure if the "next" bin exists before computing
         velflux = []
         areas = []
         ncom = 1
         for i in range(len(vel) - 1):
            velflux.append(vel[i] * flux[i])
            if bins[i+1] - bins[i] == 1:
               areas.append((flux[i]+flux[i+1]) * (vel[i+1]-vel[i]) / 2)
            else:
               ncom += 1
         int_flux = sum(areas)

         ### Add back the last entry to VelFlux then calculate the 
         ### flux-weighted mean velocity
         fvel = (sum(velflux) + vel[len(vel)-1] * flux[len(vel)-1]) / sum(flux)


         return np.array([velmin, velmax, velcov, velave, pvel, pflux, \
                          int_flux, fvel, ncom])
      else:
         return np.full(9, np.nan)




