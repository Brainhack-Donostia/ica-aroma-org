#!/usr/bin/env python
from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import str
import os
import os.path as op
import argparse
import subprocess
import shutil
from . import utils, features


def aroma_workflow(inFeat, inFile, mc, melDir, affmat, warp, outDir, dim,
                   denType, mask, TR, overwrite, generate_plots):

    # Change to script directory
    cwd = op.realpath(op.curdir)
    scriptDir = op.dirname(op.abspath(__file__))
    os.chdir(scriptDir)

    print('\n------------------------------- RUNNING ICA-AROMA ------------------------------- ')
    print('--------------- \'ICA-based Automatic Removal Of Motion Artifacts\' --------------- \n')

    # Define variables based on the type of input (i.e. Feat directory or specific input arguments), and check whether the specified files exist.
    cancel = False

    if inFeat:

        # Check whether the Feat directory exists
        if not os.path.isdir(inFeat):
            print('The specified Feat directory does not exist.')
            print('\n----------------------------- ICA-AROMA IS CANCELED -----------------------------\n')
            exit()

        # Define the variables which should be located in the Feat directory
        inFile = os.path.join(inFeat, 'filtered_func_data.nii.gz')
        mc = os.path.join(inFeat, 'mc', 'prefiltered_func_data_mcf.par')
        affmat = os.path.join(inFeat, 'reg', 'example_func2highres.mat')
        warp = os.path.join(inFeat, 'reg', 'highres2standard_warp.nii.gz')

        # Check whether these files actually exist
        if not os.path.isfile(inFile):
            print('Missing filtered_func_data.nii.gz in Feat directory.')
            cancel = True
        if not os.path.isfile(mc):
            print('Missing mc/prefiltered_func_data_mcf.mat in Feat directory.')
            cancel = True
        if not os.path.isfile(affmat):
            print('Missing reg/example_func2highres.mat in Feat directory.')
            cancel = True
        if not os.path.isfile(warp):
            print('Missing reg/highres2standard_warp.nii.gz in Feat directory.')
            cancel = True

        # Check whether a melodic.ica directory exists
        if os.path.isdir(os.path.join(inFeat, 'filtered_func_data.ica')):
            melDir = os.path.join(inFeat, 'filtered_func_data.ica')
        else:
            melDir = melDir
    else:
        inFile = inFile
        mc = mc
        affmat = affmat
        warp = warp
        melDir = melDir

        # Check whether the files exist
        if not inFile:
            print('No input file specified.')
        else:
            if not os.path.isfile(inFile):
                print('The specified input file does not exist.')
                cancel = True
        if not mc:
            print('No mc file specified.')
        else:
            if not os.path.isfile(mc):
                print('The specified mc file does does not exist.')
                cancel = True
        if affmat:
            if not os.path.isfile(affmat):
                print('The specified affmat file does not exist.')
                cancel = True
        if warp:
            if not os.path.isfile(warp):
                print('The specified warp file does not exist.')
                cancel = True

    # Parse the arguments which do not depend on whether a Feat directory has been specified
    outDir = outDir
    dim = dim
    denType = denType

    # Check if the mask exists, when specified.
    if mask:
        if not os.path.isfile(mask):
            print('The specified mask does not exist.')
            cancel = True

    # Check if the type of denoising is correctly specified, when specified
    if not (denType == 'nonaggr') and not (denType == 'aggr') and not (denType == 'both') and not (denType == 'no'):
        print('Type of denoising was not correctly specified. Non-aggressive denoising will be run.')
        denType = 'nonaggr'

    # If the criteria for file/directory specifications have not been met. Cancel ICA-AROMA.
    if cancel:
        print('\n----------------------------- ICA-AROMA IS CANCELED -----------------------------\n')
        exit()

    #------------------------------------------- PREPARE -------------------------------------------#

    # Define the FSL-bin directory
    fslDir = os.path.join(os.environ["FSLDIR"], 'bin', '')

    # Create output directory if needed
    if os.path.isdir(outDir) and overwrite is False:
        print('Output directory', outDir, """already exists.
            AROMA will not continue.
            Rerun with the -overwrite option to explicitly overwrite existing output.""")
        exit()
    elif os.path.isdir(outDir) and overwrite is True:
        print('Warning! Output directory', outDir, 'exists and will be overwritten.\n')
        shutil.rmtree(outDir)
        os.makedirs(outDir)
    else:
        os.makedirs(outDir)

    # Get TR of the fMRI data, if not specified
    if TR:
        TR = TR
    else:
        cmd = ' '.join([os.path.join(fslDir, 'fslinfo'),
                        inFile,
                        '| grep pixdim4 | awk \'{print $2}\''])
        TR = float(subprocess.getoutput(cmd))

    # Check TR
    if TR == 1:
        print('Warning! Please check whether the determined TR (of ' + str(TR) + 's) is correct!\n')
    elif TR == 0:
        print('TR is zero. ICA-AROMA requires a valid TR and will therefore exit. Please check the header, or define the TR as an additional argument.\n----------------------------- ICA-AROMA IS CANCELED -----------------------------\n')
        exit()

    # Define/create mask. Either by making a copy of the specified mask, or by creating a new one.
    new_mask = os.path.join(outDir, 'mask.nii.gz')
    if mask:
        shutil.copyfile(mask, new_mask)
    else:
        # If a Feat directory is specified, and an example_func is present use example_func to create a mask
        if inFeat and os.path.isfile(os.path.join(inFeat, 'example_func.nii.gz')):
            os.system(' '.join([os.path.join(fslDir, 'bet'),
                                os.path.join(inFeat, 'example_func.nii.gz'),
                                os.path.join(outDir, 'bet'),
                                '-f 0.3 -n -m -R']))
            os.system(' '.join(['mv',
                                os.path.join(outDir, 'bet_mask.nii.gz'),
                                mask]))
            if os.path.isfile(os.path.join(outDir, 'bet.nii.gz')):
                os.remove(os.path.join(outDir, 'bet.nii.gz'))
        else:
            if inFeat:
                print(' - No example_func was found in the Feat directory. A mask will be created including all voxels with varying intensity over time in the fMRI data. Please check!\n')
            os.system(' '.join([os.path.join(fslDir, 'fslmaths'),
                                inFile,
                                '-Tstd -bin',
                                mask]))


    #---------------------------------------- Run ICA-AROMA ----------------------------------------#

    print('Step 1) MELODIC')
    utils.runICA(fslDir, inFile, outDir, melDir, new_mask, dim, TR)

    print('Step 2) Automatic classification of the components')
    print('  - registering the spatial maps to MNI')
    melIC = os.path.join(outDir, 'melodic_IC_thr.nii.gz')
    melIC_MNI = os.path.join(outDir, 'melodic_IC_thr_MNI2mm.nii.gz')
    utils.register2MNI(fslDir, melIC, melIC_MNI, affmat, warp)

    print('  - extracting the CSF & Edge fraction features')
    edgeFract, csfFract = features.feature_spatial(fslDir, outDir, scriptDir, melIC_MNI)

    print('  - extracting the Maximum RP correlation feature')
    melmix = os.path.join(outDir, 'melodic.ica', 'melodic_mix')
    maxRPcorr = features.feature_time_series(melmix, mc)

    print('  - extracting the High-frequency content feature')
    melFTmix = os.path.join(outDir, 'melodic.ica', 'melodic_FTmix')
    HFC = features.feature_frequency(melFTmix, TR)

    print('  - classification')
    motionICs = utils.classification(outDir, maxRPcorr, edgeFract, HFC, csfFract)

    if generate_plots:
        from classification_plots import classification_plot
        classification_plot(os.path.join(outDir, 'classification_overview.txt'),
                            outDir)


    if (denType != 'no'):
        print('Step 3) Data denoising')
        utils.denoising(fslDir, inFile, outDir, melmix, denType, motionICs)

    # Revert to old directory
    os.chdir(cwd)

    print('\n----------------------------------- Finished -----------------------------------\n')
