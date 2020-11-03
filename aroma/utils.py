"""Utility functions for ICA-AROMA.
"""
import os
import os.path as op
import subprocess
from glob import glob

import nibabel as nib
import numpy as np


def runICA(fslDir, inFile, outDir, melDirIn, mask, dim, TR):
    """Run MELODIC and merge the mixture modeled thresholded
    ICs into a single 4D nifti file.

    Parameters
    ----------
    fslDir : str
        Full path of the bin-directory of FSL
    inFile : str
        Full path to the fMRI data file (nii.gz) on which MELODIC
        should be run
    outDir : str
        Full path of the output directory
    melDirIn : str
        Full path of the MELODIC directory in case it has been run
        before, otherwise define empty string
    mask : str
        Full path of the mask to be applied during MELODIC
    dim : int
        Dimensionality of ICA
    TR : float
        TR (in seconds) of the fMRI data

    Output
    ------
    melodic.ica/: MELODIC directory
    melodic_IC_thr.nii.gz: Merged file containing the mixture modeling
                           thresholded Z-statistical maps located in
                           melodic.ica/stats/
    """
    # Define the 'new' MELODIC directory and predefine some associated files
    melDir = op.join(outDir, 'melodic.ica')
    melIC = op.join(melDir, 'melodic_IC.nii.gz')
    melICmix = op.join(melDir, 'melodic_mix')
    melICthr = op.join(outDir, 'melodic_IC_thr.nii.gz')

    # When a MELODIC directory is specified,
    # check whether all needed files are present.
    # Otherwise... run MELODIC again
    if (op.isfile(op.join(melDirIn, 'melodic_IC.nii.gz')) and
            op.isfile(op.join(melDirIn, 'melodic_FTmix')) and
            op.isfile(op.join(melDirIn, 'melodic_mix'))):
        print('  - The existing/specified MELODIC directory will be used.')

        # If a 'stats' directory is present (contains thresholded spatial maps)
        # create a symbolic link to the MELODIC directory.
        # Otherwise create specific links and
        # run mixture modeling to obtain thresholded maps.
        if op.isdir(op.join(melDirIn, 'stats')):
            os.symlink(melDirIn, melDir)
        else:
            print("  - The MELODIC directory does not contain the required "
                  "'stats' folder. Mixture modeling on the Z-statistical "
                  "maps will be run.")

            # Create symbolic links to the items in the specified melodic
            # directory
            os.makedirs(melDir)
            for item in os.listdir(melDirIn):
                os.symlink(op.join(melDirIn, item),
                           op.join(melDir, item))

            # Run mixture modeling
            melodic_command = ("{0} --in={1} --ICs={1} --mix={2} --outdir={3} "
                               "--0stats --mmthresh=0.5").format(
                                    op.join(fslDir, 'melodic'),
                                    melIC,
                                    melICmix,
                                    melDir,
                               )
            os.system(melodic_command)
    else:
        # If a melodic directory was specified, display that it did not
        # contain all files needed for ICA-AROMA (or that the directory
        # does not exist at all)
        if melDirIn:
            if not op.isdir(melDirIn):
                print('  - The specified MELODIC directory does not exist. '
                      'MELODIC will be run seperately.')
            else:
                print('  - The specified MELODIC directory does not contain '
                      'the required files to run ICA-AROMA. MELODIC will be '
                      'run seperately.')

        # Run MELODIC
        melodic_command = ("{0} --in={1} --outdir={2} --mask={3} --dim={4} "
                           "--Ostats --nobet --mmthresh=0.5 --report "
                           "--tr={5}").format(
                               op.join(fslDir, 'melodic'),
                               inFile,
                               melDir,
                               mask,
                               dim,
                               TR
                           )
        os.system(melodic_command)

    # Get number of components
    melIC_img = nib.load(melIC)
    nrICs = melIC_img.shape[3]

    # Merge mixture modeled thresholded spatial maps. Note! In case that
    # mixture modeling did not converge, the file will contain two spatial
    # maps. The latter being the results from a simple null hypothesis test.
    # In that case, this map will have to be used (first one will be empty).
    for i in range(1, nrICs + 1):
        # Define thresholded zstat-map file
        zTemp = op.join(melDir, 'stats', 'thresh_zstat{0}.nii.gz'.format(i))
        cmd = "{0} {1} | grep dim4 | head -n1 | awk '{{print $2}}'".format(
            op.join(fslDir, 'fslinfo'),
            zTemp
        )
        lenIC = int(float(subprocess.getoutput(cmd)))

        # Define zeropad for this IC-number and new zstat file
        cmd = ' '.join([op.join(fslDir, 'zeropad'),
                        str(i),
                        '4'])
        ICnum = subprocess.getoutput(cmd)
        zstat = op.join(outDir, 'thr_zstat' + ICnum)

        # Extract last spatial map within the thresh_zstat file
        os.system(' '.join([op.join(fslDir, 'fslroi'),
                            zTemp,      # input
                            zstat,      # output
                            str(lenIC - 1),   # first frame
                            '1']))      # number of frames

    # Merge and subsequently remove all mixture modeled Z-maps within the
    # output directory
    merge_command = "{0} -t {1} {2}".format(
        op.join(fslDir, 'fslmerge'),
        melICthr,
        op.join(outDir, 'thr_zstat????.nii.gz')
    )
    os.system(merge_command)  # inputs

    component_images = glob(
        op.join(outDir, 'thr_zstat[0-9][0-9][0-9][0-9].nii.gz')
    )
    for f in component_images:
        os.remove(f)

    # Apply the mask to the merged file (in case a melodic-directory was
    # predefined and run with a different mask)
    math_command = "{0} {1} -mas {2} {3}".format(
        op.join(fslDir, 'fslmaths'),
        melICthr,
        mask,
        melICthr,
    )
    os.system(math_command)


def register2MNI(fslDir, inFile, outFile, affmat, warp):
    """Register an image (or time-series of images) to MNI152 T1 2mm.

    If no affmat is defined, it only warps (i.e. it assumes that the data has
    been registered to the structural scan associated with the warp-file
    already). If no warp is defined either, it only resamples the data to 2mm
    isotropic if needed (i.e. it assumes that the data has been registered to
    a MNI152 template). In case only an affmat file is defined, it assumes that
    the data has to be linearly registered to MNI152 (i.e. the user has a
    reason not to use non-linear registration on the data).

    Parameters
    ----------
    fslDir : str
        Full path of the bin-directory of FSL
    inFile : str
        Full path to the data file (nii.gz) which has to be registerd to
        MNI152 T1 2mm
    outFile : str
        Full path of the output file
    affmat : str
        Full path of the mat file describing the linear registration (if data
        is still in native space)
    warp : str
        Full path of the warp file describing the non-linear registration (if
        data has not been registered to MNI152 space yet)

    Output
    ------
    melodic_IC_mm_MNI2mm.nii.gz : merged file containing the mixture modeling
                                  thresholded Z-statistical maps registered to
                                  MNI152 2mm
    """
    # Define the MNI152 T1 2mm template
    fslnobin = fslDir.rsplit('/', 2)[0]
    ref = op.join(fslnobin, 'data', 'standard', 'MNI152_T1_2mm_brain.nii.gz')

    # If the no affmat- or warp-file has been specified, assume that the data
    # is already in MNI152 space. In that case only check if resampling to
    # 2mm is needed
    if affmat and warp:
        in_img = nib.load(inFile)
        # Get 3D voxel size
        pixdim1, pixdim2, pixdim3 = in_img.header.get_zooms()[:3]

        # If voxel size is not 2mm isotropic, resample the data, otherwise
        # copy the file
        if (pixdim1 != 2) or (pixdim2 != 2) or (pixdim3 != 2):
            os.system(' '.join([op.join(fslDir, 'flirt'),
                                ' -ref ' + ref,
                                ' -in ' + inFile,
                                ' -out ' + outFile,
                                ' -applyisoxfm 2 -interp trilinear']))
        else:
            os.copyfile(inFile, outFile)

    # If only a warp-file has been specified, assume that the data has already
    # been registered to the structural scan. In that case apply the warping
    # without a affmat
    elif affmat and warp:
        # Apply warp
        os.system(' '.join([op.join(fslDir, 'applywarp'),
                            '--ref=' + ref,
                            '--in=' + inFile,
                            '--out=' + outFile,
                            '--warp=' + warp,
                            '--interp=trilinear']))

    # If only a affmat-file has been specified perform affine registration to
    # MNI
    elif affmat and warp:
        os.system(' '.join([op.join(fslDir, 'flirt'),
                            '-ref ' + ref,
                            '-in ' + inFile,
                            '-out ' + outFile,
                            '-applyxfm -init ' + affmat,
                            '-interp trilinear']))

    # If both a affmat- and warp-file have been defined, apply the warping
    # accordingly
    else:
        os.system(' '.join([op.join(fslDir, 'applywarp'),
                            '--ref=' + ref,
                            '--in=' + inFile,
                            '--out=' + outFile,
                            '--warp=' + warp,
                            '--premat=' + affmat,
                            '--interp=trilinear']))


def cross_correlation(a, b):
    """Perform cross-correlations between columns of two matrices.

    Parameters
    ----------
    a : (M x X) array_like
        First array to cross-correlate
    b : (N x X) array_like
        Second array to cross-correlate

    Returns
    -------
    correlations : (M x N) array_like
        Cross-correlations of columns of a against columns of b.
    """
    assert a.ndim == b.ndim == 2
    _, ncols_a = a.shape
    # nb variables in columns rather than rows hence transpose
    # extract just the cross terms between cols in a and cols in b
    return np.corrcoef(a.T, b.T)[:ncols_a, ncols_a:]


def classification(outDir, maxRPcorr, edgeFract, HFC, csfFract):
    """Classify a set of components as motion or non-motion based on four
    features; maximum RP correlation, high-frequency content, edge-fraction
    and CSF-fraction.

    Parameters
    ----------
    outDir : str
        Full path of the output directory
    maxRPcorr : (C,) array_like
        Array of the 'maximum RP correlation' feature scores of the components
    edgeFract : (C,) array_like
        Array of the 'edge fraction' feature scores of the components
    HFC : (C,) array_like
        Array of the 'high-frequency content' feature scores of the components
    csfFract : (C,) array_like
        Array of the 'CSF fraction' feature scores of the components

    Returns
    -------
    motionICs : array_like
        Array containing the indices of the components identified as motion
        components

    Output
    ------
    classified_motion_ICs.txt : A text file containing the indices of the
                                components identified as motion components
    """
    # Classify the ICs as motion or non-motion

    # Define criteria needed for classification (thresholds and
    # hyperplane-parameters)
    thr_csf = 0.10
    thr_HFC = 0.35
    hyp = np.array([-19.9751070082159, 9.95127547670627, 24.8333160239175])

    # Project edge & maxRPcorr feature scores to new 1D space
    x = np.array([maxRPcorr, edgeFract])
    proj = hyp[0] + np.dot(x.T, hyp[1:])

    # Classify the ICs
    motionICs = np.squeeze(
        np.array(
            np.where(
                (proj > 0) +
                (csfFract > thr_csf) +
                (HFC > thr_HFC)
            )
        )
    )

    # Put the feature scores in a text file
    np.savetxt(op.join(outDir, 'feature_scores.txt'),
               np.vstack((maxRPcorr, edgeFract, HFC, csfFract)).T)

    # Put the indices of motion-classified ICs in a text file
    with open(op.join(outDir, 'classified_motion_ICs.txt'), 'w') as fo:
        if motionICs.size > 1:
            fo.write(','.join(['{:.0f}'.format(num) for num in
                               (motionICs + 1)]))
        elif motionICs.size == 1:
            fo.write('{:.0f}'.format(motionICs + 1))

    # Create a summary overview of the classification
    with open(op.join(outDir, 'classification_overview.txt'), 'w') as fo:
        fo.write('\t'.join(['IC',
                            'Motion/noise',
                            'maximum RP correlation',
                            'Edge-fraction',
                            'High-frequency content',
                            'CSF-fraction']))
        fo.write('\n')
        for i in range(0, len(csfFract)):
            if (proj[i] > 0) or (csfFract[i] > thr_csf) or (HFC[i] > thr_HFC):
                classif = "True"
            else:
                classif = "False"
            fo.write('\t'.join(['{:d}'.format(i + 1),
                                classif,
                                '{:.2f}'.format(maxRPcorr[i]),
                                '{:.2f}'.format(edgeFract[i]),
                                '{:.2f}'.format(HFC[i]),
                                '{:.2f}'.format(csfFract[i])]))
            fo.write('\n')

    return motionICs


def denoising(fslDir, inFile, outDir, melmix, denType, denIdx):
    """Classify the ICs based on the four features: maximum RP correlation,
    high-frequency content, edge-fraction and CSF-fraction.

    Parameters
    ----------
    fslDir : str
        Full path of the bin-directory of FSL
    inFile : str
        Full path to the data file (nii.gz) which has to be denoised
    outDir : str
        Full path of the output directory
    melmix : str
        Full path of the melodic_mix text file
    denType : {"aggr", "nonaggr", "both"}
        Type of requested denoising ('aggr': aggressive, 'nonaggr':
        non-aggressive, 'both': both aggressive and non-aggressive
    denIdx : array_like
        Index of the components that should be regressed out

    Output
    ------
    denoised_func_data_<denType>.nii.gz : The denoised fMRI data
    """
    # Check if denoising is needed (i.e. are there motion components?)
    check = denIdx.size > 0

    if check == 1:
        # Put IC indices into a char array
        if denIdx.size == 1:
            denIdxStrJoin = str(denIdx + 1)
        else:
            denIdxStrJoin = ','.join([str(i + 1) for i in denIdx])

        # Non-aggressive denoising of the data using fsl_regfilt
        # (partial regression), if requested
        if denType in ('nonaggr', 'both'):
            regfilt_command = ("{0} --in={1} --design={2} --filter='{3}' "
                               "--out={4}").format(
                                   op.join(fslDir, 'fsl_regfilt'),
                                   inFile,
                                   melmix,
                                   denIdxStrJoin,
                                   op.join(
                                       outDir,
                                       'denoised_func_data_nonaggr.nii.gz'
                                   )
                               )
            os.system(regfilt_command)

        # Aggressive denoising of the data using fsl_regfilt (full regression)
        if denType in ('aggr', 'both'):
            regfilt_command = ("{0} --in={1} --design={2} --filter='{3}' "
                               "--out={4} -a").format(
                                   op.join(fslDir, 'fsl_regfilt'),
                                   inFile,
                                   melmix,
                                   denIdxStrJoin,
                                   op.join(
                                       outDir,
                                       'denoised_func_data_aggr.nii.gz'
                                   )
                               )
            os.system(regfilt_command)
    else:
        print("  - None of the components were classified as motion, so no "
              "denoising is applied (a symbolic link to the input file will "
              "be created).")
        if denType in ('nonaggr', 'both'):
            os.symlink(
                inFile,
                op.join(outDir, 'denoised_func_data_nonaggr.nii.gz')
            )

        if denType in ('aggr', 'both'):
            os.symlink(
                inFile,
                op.join(outDir, 'denoised_func_data_aggr.nii.gz')
            )


def get_resource_path():
    """
    Returns the path to general resources, terminated with separator. Resources
    are kept outside package folder in "datasets".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return op.abspath(op.join(op.dirname(__file__), "resources") + op.sep)
