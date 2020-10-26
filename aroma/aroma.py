#!/usr/bin/env python
import os
import os.path as op
import shutil

import nibabel as nib

from . import utils, features


def aroma_workflow(inFeat, inFile, mc, melDir, affmat, warp, outDir, dim,
                   denType, mask, TR, overwrite, generate_plots):
    # Change to script directory
    cwd = op.realpath(op.curdir)
    scriptDir = op.dirname(op.abspath(__file__))
    os.chdir(scriptDir)

    print(
        "\n------------------------ RUNNING ICA-AROMA ------------------------"
    )
    print(
        "-------- 'ICA-based Automatic Removal Of Motion Artifacts' --------\n"
    )

    # Define variables based on the type of input (i.e. Feat directory or
    # specific input arguments), and check whether the specified files exist.
    cancel = False

    if inFeat:
        inFeat = inFeat

        # Check whether the Feat directory exists
        if not op.isdir(inFeat):
            print("The specified Feat directory does not exist.")
            print(
                "\n----------------- ICA-AROMA IS CANCELED -----------------\n"
            )
            return

        # Define the variables which should be located in the Feat directory
        inFile = op.join(inFeat, "filtered_func_data.nii.gz")
        mc = op.join(inFeat, "mc", "prefiltered_func_data_mcf.par")
        affmat = op.join(inFeat, "reg", "example_func2highres.mat")
        warp = op.join(inFeat, "reg", "highres2standard_warp.nii.gz")

        # Check whether these files actually exist
        if not op.isfile(inFile):
            print("Missing filtered_func_data.nii.gz in Feat directory.")
            cancel = True
        if not op.isfile(mc):
            print("Missing mc/prefiltered_func_data_mcf.mat in Feat "
                  "directory.")
            cancel = True
        if not op.isfile(affmat):
            print("Missing reg/example_func2highres.mat in Feat directory.")
            cancel = True
        if not op.isfile(warp):
            print("Missing reg/highres2standard_warp.nii.gz in Feat "
                  "directory.")
            cancel = True

        # Check whether a melodic.ica directory exists
        if op.isdir(op.join(inFeat, "filtered_func_data.ica")):
            melDir = op.join(inFeat, "filtered_func_data.ica")
    else:
        # Check whether the files exist
        if not inFile:
            print("No input file specified.")
        else:
            if not op.isfile(inFile):
                print("The specified input file does not exist.")
                cancel = True

        if not mc:
            print("No mc file specified.")
        else:
            if not op.isfile(mc):
                print("The specified mc file does does not exist.")
                cancel = True

        if affmat:
            if not op.isfile(affmat):
                print("The specified affmat file does not exist.")
                cancel = True

        if warp:
            if not op.isfile(warp):
                print("The specified warp file does not exist.")
                cancel = True

    # Check if the mask exists, when specified.
    if mask:
        if not op.isfile(mask):
            print("The specified mask does not exist.")
            cancel = True

    # Check if the type of denoising is correctly specified, when specified
    if denType not in ("nonaggr", "aggr", "both", "no"):
        print(
            "Type of denoising was not correctly specified. Non-aggressive "
            "denoising will be run."
        )
        denType = "nonaggr"

    # If the criteria for file/directory specifications have not been met.
    # Cancel ICA-AROMA.
    if cancel:
        print(
            "\n------------------- ICA-AROMA IS CANCELED -------------------\n"
        )
        return

    # ------------------------------------------- PREPARE --------------------#

    # Define the FSL-bin directory
    fslDir = op.join(os.environ["FSLDIR"], "bin", "")

    # Create output directory if needed
    if op.isdir(outDir) and overwrite is False:
        print(
            "Output directory",
            outDir,
            """already exists.
            AROMA will not continue.
            Rerun with the -overwrite option to explicitly overwrite
            existing output.""",
        )
        return
    elif op.isdir(outDir) and overwrite is True:
        print("Warning! Output directory {} exists and will be overwritten."
              "\n".format(outDir))
        shutil.rmtree(outDir)
        os.makedirs(outDir)
    else:
        os.makedirs(outDir)

    # Get TR of the fMRI data, if not specified
    if not TR:
        in_img = nib.load(inFile)
        TR = in_img.header.get_zooms()[3]

    # Check TR
    if TR == 1:
        print(
            "Warning! Please check whether the determined TR (of "
            + str(TR)
            + "s) is correct!\n"
        )
    elif TR == 0:
        print(
            "TR is zero. ICA-AROMA requires a valid TR and will therefore "
            "exit. Please check the header, or define the TR as an additional "
            "argument.\n"
            "-------------- ICA-AROMA IS CANCELED ------------\n"
        )
        return

    # Define/create mask. Either by making a copy of the specified mask, or by
    # creating a new one.
    new_mask = op.join(outDir, "mask.nii.gz")
    if mask:
        shutil.copyfile(mask, new_mask)
    elif inFeat and op.isfile(op.join(inFeat, "example_func.nii.gz")):
        # If a Feat directory is specified, and an example_func is present use
        # example_func to create a mask
        bet_command = "{0} {1} {2} -f 0.3 -n -m -R".format(
            op.join(fslDir, "bet"),
            op.join(inFeat, "example_func.nii.gz"),
            op.join(outDir, "bet"),
        )
        os.system(bet_command)
        os.rename(op.join(outDir, "bet_mask.nii.gz"), new_mask)
        if op.isfile(op.join(outDir, "bet.nii.gz")):
            os.remove(op.join(outDir, "bet.nii.gz"))
    else:
        if inFeat:
            print(
                " - No example_func was found in the Feat directory. "
                "A mask will be created including all voxels with varying "
                "intensity over time in the fMRI data. Please check!\n"
            )
        math_command = "{0} {1} -Tstd -bin {2}".format(
            op.join(fslDir, "fslmaths"),
            inFile,
            new_mask
        )
        os.system(math_command)

    # Run ICA-AROMA
    print("Step 1) MELODIC")
    utils.runICA(fslDir, inFile, outDir, melDir, new_mask, dim, TR)

    print("Step 2) Automatic classification of the components")
    print("  - registering the spatial maps to MNI")
    melIC = op.join(outDir, "melodic_IC_thr.nii.gz")
    melIC_MNI = op.join(outDir, "melodic_IC_thr_MNI2mm.nii.gz")
    utils.register2MNI(fslDir, melIC, melIC_MNI, affmat, warp)

    print("  - extracting the CSF & Edge fraction features")
    edgeFract, csfFract = features.feature_spatial(
        fslDir, outDir, scriptDir, melIC_MNI
    )

    print("  - extracting the Maximum RP correlation feature")
    melmix = op.join(outDir, "melodic.ica", "melodic_mix")
    maxRPcorr = features.feature_time_series(melmix, mc)

    print("  - extracting the High-frequency content feature")
    melFTmix = op.join(outDir, "melodic.ica", "melodic_FTmix")
    HFC = features.feature_frequency(melFTmix, TR)

    print("  - classification")
    motionICs = utils.classification(
        outDir,
        maxRPcorr,
        edgeFract,
        HFC,
        csfFract
    )

    if generate_plots:
        from . import plotting

        plotting.classification_plot(
            op.join(outDir, "classification_overview.txt"),
            outDir
        )

    if denType != "no":
        print("Step 3) Data denoising")
        utils.denoising(fslDir, inFile, outDir, melmix, denType, motionICs)

    # Revert to old directory
    os.chdir(cwd)

    print("Finished")
