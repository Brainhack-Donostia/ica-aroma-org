"""
Functions to calculate ICA-AROMA features for component classification.
"""
import os

import nibabel as nib
import numpy as np
from nilearn import image, masking

from .utils import cross_correlation, get_resource_path


def feature_time_series(melmix, mc):
    """Extract maximum motion parameter correlation feature scores from
    component time series.

    This function determines the maximum robust correlation of each component
    time series with a model of 72 realignment parameters.

    Parameters
    ----------
    melmix : str
        Full path of the melodic_mix text file
    mc : str
        Full path of the text file containing the realignment parameters

    Returns
    -------
    maxRPcorr : array_like
        Array of the maximum RP correlation feature scores for the components
        of the melodic_mix file
    """
    # Read melodic mix file (IC time-series), subsequently define a set of
    # squared time-series
    mix = np.loadtxt(melmix)

    # Read motion parameter file
    rp6 = np.loadtxt(mc)

    # Determine the derivatives of the RPs (add zeros at time-point zero)
    _, nparams = rp6.shape
    rp6_der = np.vstack((
        np.zeros(nparams),
        np.diff(rp6, axis=0)
    ))

    # Create an RP-model including the RPs and its derivatives
    rp12 = np.hstack((rp6, rp6_der))

    # Add the squared RP-terms to the model
    # NOTE: The above comment existed, but this step was **missing**!!
    rp24 = np.hstack((rp12, rp12 ** 2))

    # add the fw and bw shifted versions
    rp12_1fw = np.vstack((
        np.zeros(2 * nparams),
        rp12[:-1]
    ))
    rp12_1bw = np.vstack((
        rp12[1:],
        np.zeros(2 * nparams)
    ))
    rp_model = np.hstack((rp24, rp12_1fw, rp12_1bw))

    # Determine the maximum correlation between RPs and IC time-series
    nsplits = 1000
    nmixrows, nmixcols = mix.shape
    nrows_to_choose = int(round(0.9 * nmixrows))

    # Max correlations for multiple splits of the dataset (for a robust
    # estimate)
    max_correls = np.empty((nsplits, nmixcols))
    for i in range(nsplits):
        # Select a random subset of 90% of the dataset rows
        # (*without* replacement)
        chosen_rows = np.random.choice(a=range(nmixrows),
                                       size=nrows_to_choose,
                                       replace=False)

        # Combined correlations between RP and IC time-series, squared and
        # non squared
        correl_nonsquared = cross_correlation(mix[chosen_rows],
                                              rp_model[chosen_rows])
        correl_squared = cross_correlation(mix[chosen_rows]**2,
                                           rp_model[chosen_rows]**2)
        correl_both = np.hstack((correl_squared, correl_nonsquared))

        # Maximum absolute temporal correlation for every IC
        max_correls[i] = np.abs(correl_both).max(axis=1)

    # Feature score is the mean of the maximum correlation over all the random
    # splits
    # Avoid propagating occasional nans that arise in artificial test cases
    maxRPcorr = np.nanmean(max_correls, axis=0)
    return maxRPcorr


def feature_frequency(melFTmix, TR):
    """Extract the high-frequency content feature scores.

    This function determines the frequency, as fraction of the Nyquist
    frequency, at which the higher and lower frequencies explain half
    of the total power between 0.01Hz and Nyquist.

    Parameters
    ----------
    melFTmix : str
        Full path of the melodic_FTmix text file
    TR : float
        TR (in seconds) of the fMRI data

    Returns
    -------
    HFC : array_like
        Array of the HFC ('High-frequency content') feature scores
        for the components of the melodic_FTmix file
    """
    # Determine sample frequency
    Fs = 1 / TR

    # Determine Nyquist-frequency
    Ny = Fs / 2

    # Load melodic_FTmix file
    FT = np.loadtxt(melFTmix)

    # Determine which frequencies are associated with every row in the
    # melodic_FTmix file  (assuming the rows range from 0Hz to Nyquist)
    f = Ny * (np.array(list(range(1, FT.shape[0] + 1)))) / (FT.shape[0])

    # Only include frequencies higher than 0.01Hz
    fincl = np.squeeze(np.array(np.where(f > 0.01)))
    FT = FT[fincl, :]
    f = f[fincl]

    # Set frequency range to [0-1]
    f_norm = (f - 0.01) / (Ny - 0.01)

    # For every IC; get the cumulative sum as a fraction of the total sum
    fcumsum_fract = np.cumsum(FT, axis=0) / np.sum(FT, axis=0)

    # Determine the index of the frequency with the fractional cumulative sum
    # closest to 0.5
    idx_cutoff = np.argmin(np.abs(fcumsum_fract - 0.5), axis=0)

    # Now get the fractions associated with those indices index, these are the
    # final feature scores
    HFC = f_norm[idx_cutoff]

    # Return feature score
    return HFC


def feature_spatial(fslDir, tempDir, aromaDir, melIC):
    """Extract the spatial feature scores.

    For each IC it determines the fraction of the mixture modeled thresholded
    Z-maps respectively located within the CSF or at the brain edges,
    using predefined standardized masks.

    Parameters
    ----------
    fslDir : str
        Full path of the bin-directory of FSL
    tempDir : str
        Full path of a directory where temporary files can be stored
        (called 'temp_IC.nii.gz')
    aromaDir : str
        Full path of the ICA-AROMA directory, containing the mask-files
        (mask_edge.nii.gz, mask_csf.nii.gz & mask_out.nii.gz)
    melIC : str
        Full path of the nii.gz file containing mixture-modeled threholded
        (p>0.5) Z-maps, registered to the MNI152 2mm template

    Returns
    -------
    edgeFract : array_like
        Array of the edge fraction feature scores for the components of the
        melIC file
    csfFract : array_like
        Array of the CSF fraction feature scores for the components of the
        melIC file
    """
    # Get the number of ICs
    melIC_img = nib.load(melIC)
    numICs = melIC_img.shape[3]

    # Loop over ICs
    edgeFract = np.zeros(numICs)
    csfFract = np.zeros(numICs)
    for i in range(numICs):
        # Extract IC from the merged melodic_IC_thr2MNI2mm file
        tempIC = image.index_img(melIC, i)

        # Change to absolute Z-values
        tempIC = image.math_img("np.abs(img)", img=tempIC)

        # Get sum of Z-values within the total Z-map (calculate via the mean
        # and number of non-zero voxels)
        tempICdata = tempIC.get_fdata()
        totVox = np.sum(tempICdata != 0)  # number of nonzero voxels in image

        if totVox != 0:
            totMean = np.mean(tempICdata[tempICdata != 0])
        else:
            print("\t- The spatial map of component {} is empty. "
                  "Please check!".format(i+1))
            totMean = 0

        totSum = totMean * totVox

        # Get sum of Z-values of the voxels located within the CSF
        # (calculate via the mean and number of non-zero voxels)
        csf_mask = os.path.join(get_resource_path(), "mask_csf.nii.gz")
        csf_data = masking.apply_mask(tempIC, csf_mask)
        csfVox = np.sum(csf_data != 0)  # number of nonzero voxels in mask

        if not (csfVox == 0):
            csfMean = np.mean(csf_data[csf_data != 0])
        else:
            csfMean = 0

        csfSum = csfMean * csfVox

        # Get sum of Z-values of the voxels located within the Edge
        # (calculate via the mean and number of non-zero voxels)
        edge_mask = os.path.join(get_resource_path(), "mask_edge.nii.gz")
        edge_data = masking.apply_mask(tempIC, edge_mask)
        edgeVox = np.sum(edge_data != 0)  # number of nonzero voxels in mask

        if not (edgeVox == 0):
            edgeMean = np.mean(edge_data[edge_data != 0])
        else:
            edgeMean = 0

        edgeSum = edgeMean * edgeVox

        # Get sum of Z-values of the voxels located outside the brain
        # (calculate via the mean and number of non-zero voxels)
        out_mask = os.path.join(get_resource_path(), "mask_out.nii.gz")
        out_data = masking.apply_mask(tempIC, out_mask)
        outVox = np.sum(out_data != 0)  # number of nonzero voxels in mask

        if not (outVox == 0):
            outMean = np.mean(out_data[out_data != 0])
        else:
            outMean = 0

        outSum = outMean * outVox

        # Determine edge and CSF fraction
        if not (totSum == 0):
            edgeFract[i] = (outSum + edgeSum) / (totSum - csfSum)
            csfFract[i] = csfSum / totSum
        else:
            edgeFract[i] = 0
            csfFract[i] = 0

    # Return feature scores
    return edgeFract, csfFract
