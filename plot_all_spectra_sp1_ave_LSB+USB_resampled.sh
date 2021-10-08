bands="Band_1a Band_1b Band_2a Band_2b Band_3a Band_3b Band_4a Band_4b Band_5a Band_5b Band_6a Band_6b Band_7a Band_7b"

for band in $bands
do
    echo "Plotting $band ..."
    cd $band/Spectra
    python ../../plot_spectra_sp1_ave_LSB+USB_resampled.py *.sp1.ave.resampled.dat
    cd ../..
done
exit 0

