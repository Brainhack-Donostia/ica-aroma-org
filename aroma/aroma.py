"""The core workflow for AROMA."""
import os
import os.path as op
import shutil

import nibabel as nib

from . import utils, features


def aroma_workflow(
    out_dir,
    in_feat=None,
    in_file=None,
    mc=None,
    mel_dir=None,
    affmat=None,
    warp=None,
    dim=0,
    den_type="nonaggr",
    mask=None,
    TR=None,
    overwrite=False,
    generate_plots=True,
    csf=None,
):
    """Run the AROMA workflow.

    Parameters
    ----------
    in_feat
    """
    print("\n------------------------ RUNNING ICA-AROMA ------------------------")
    print("-------- 'ICA-based Automatic Removal Of Motion Artifacts' --------\n")
    if in_feat and in_file:
        raise ValueError("Only one of 'in_feat' and 'in_file' may be provided.")

    if in_feat and (mc or affmat or warp or mask):
        raise ValueError(
            "Arguments 'mc', 'affmat', 'warp', and 'mask' are incompatible "
            "with argument 'in_feat'."
        )

    # Define variables based on the type of input (i.e. Feat directory or
    # specific input arguments), and check whether the specified files exist.
    if in_feat:
        # Check whether the Feat directory exists
        if not op.isdir(in_feat):
            raise Exception("The specified FEAT directory does not exist.")

        # Define the variables which should be located in the Feat directory
        in_file = op.join(in_feat, "filtered_func_data.nii.gz")
        mc = op.join(in_feat, "mc", "prefiltered_func_data_mcf.par")
        affmat = op.join(in_feat, "reg", "example_func2highres.mat")
        warp = op.join(in_feat, "reg", "highres2standard_warp.nii.gz")

        # Check whether these files actually exist
        if not op.isfile(in_file):
            raise Exception("Missing filtered_func_data.nii.gz in Feat directory.")

        if not op.isfile(mc):
            raise Exception(
                "Missing mc/prefiltered_func_data_mcf.mat in Feat directory."
            )

        if not op.isfile(affmat):
            raise Exception("Missing reg/example_func2highres.mat in Feat directory.")

        if not op.isfile(warp):
            raise Exception(
                "Missing reg/highres2standard_warp.nii.gz in Feat directory."
            )

        # Check whether a melodic.ica directory exists
        if op.isdir(op.join(in_feat, "filtered_func_data.ica")):
            mel_dir = op.join(in_feat, "filtered_func_data.ica")
    else:
        # Check whether the files exist
        if not in_file:
            print("No input file specified.")
        elif not op.isfile(in_file):
            raise Exception("The specified input file does not exist.")

        if not mc:
            print("No mc file specified.")
        elif not op.isfile(mc):
            raise Exception("The specified mc file does does not exist.")

        if affmat and not op.isfile(affmat):
            raise Exception("The specified affmat file does not exist.")

        if warp and not op.isfile(warp):
            raise Exception("The specified warp file does not exist.")

    # Check if the mask exists, when specified.
    if mask and not op.isfile(mask):
        raise Exception("The specified mask does not exist.")

    # Check if the type of denoising is correctly specified, when specified
    if den_type not in ("nonaggr", "aggr", "both", "no"):
        print(
            "Type of denoising was not correctly specified. Non-aggressive "
            "denoising will be run."
        )
        den_type = "nonaggr"

    # Prepare

    # Define the FSL-bin directory
    fsl_dir = op.join(os.environ["FSLDIR"], "bin", "")

    # Create output directory if needed
    if op.isdir(out_dir) and not overwrite:
        print(
            "Output directory",
            out_dir,
            """already exists.
            AROMA will not continue.
            Rerun with the -overwrite option to explicitly overwrite
            existing output.""",
        )
        return
    elif op.isdir(out_dir) and overwrite:
        print(
            "Warning! Output directory {} exists and will be overwritten."
            "\n".format(out_dir)
        )
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)
    else:
        os.makedirs(out_dir)

    # Get TR of the fMRI data, if not specified
    if not TR:
        in_img = nib.load(in_file)
        TR = in_img.header.get_zooms()[3]

    # Check TR
    if TR == 1:
        print(
            "Warning! Please check whether the determined TR (of "
            + str(TR)
            + "s) is correct!\n"
        )
    elif TR == 0:
        raise Exception(
            "TR is zero. ICA-AROMA requires a valid TR and will therefore "
            "exit. Please check the header, or define the TR as an additional "
            "argument.\n"
            "-------------- ICA-AROMA IS CANCELED ------------\n"
        )

    # Define/create mask. Either by making a copy of the specified mask, or by
    # creating a new one.
    masks = utils.derive_masks(in_file, csf=None)

    # Run ICA-AROMA
    print("Step 1) MELODIC")
    component_maps, mixing, mixing_FT = utils.runICA(
        fsl_dir, in_file, out_dir, mel_dir, masks["brain"], dim, TR
    )

    print("Step 2) Automatic classification of the components")
    print("  - registering the spatial maps to MNI")
    mel_IC_MNI = op.join(out_dir, "melodic_IC_thr_MNI2mm.nii.gz")
    utils.register2MNI(fsl_dir, component_maps, mel_IC_MNI, affmat, warp)

    print("  - extracting the CSF & Edge fraction features")
    edge_fract, csf_fract = features.feature_spatial(mel_IC_MNI, masks)

    print("  - extracting the Maximum RP correlation feature")
    max_RP_corr = features.feature_time_series(mixing, mc)

    print("  - extracting the High-frequency content feature")
    HFC = features.feature_frequency(mixing_FT, TR)

    print("  - classification")
    motion_ICs = utils.classification(out_dir, max_RP_corr, edge_fract, HFC, csf_fract)

    if generate_plots:
        from . import plotting

        plotting.classification_plot(
            op.join(out_dir, "classification_overview.txt"), out_dir
        )

    if den_type != "no":
        print("Step 3) Data denoising")
        utils.denoising(fsl_dir, in_file, out_dir, mixing, den_type, motion_ICs)

    print("Finished")
