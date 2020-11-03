import numpy as np
import os
import pandas as pd
import subprocess
from os.path import join, split, isfile
from argparse import Namespace

from aroma.aroma import aroma_workflow

import pytest


def test_integration(skip_integration, nilearn_data):
    if skip_integration:
        pytest.skip('Skipping integration test')

    # Obtain test path
    test_path, _ = split(nilearn_data.func[0])

    # Create output path
    out_path = join(test_path, 'out')

    # Read confounds
    confounds = pd.read_csv(nilearn_data.confounds[0], sep='\t')

    # Extract motion parameters from confounds
    mc = confounds[["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]]
    mc_path = join(test_path, 'mc.tsv')
    mc.to_csv(mc_path, sep='\t', index=False, header=None)

    # Run AROMA
    aroma_workflow(TR=2, affmat='', denType='nonaggr', dim=0, generate_plots=False, inFeat=None, inFile=nilearn_data.func[0],
                   mask='', mc=mc_path, melDir='', outDir=out_path, overwrite=True, warp='')

    # Make sure files are generated
    assert isfile(join(out_path, 'classification_overview.txt'))
    assert isfile(join(out_path, 'classified_motion_ICs.txt'))
    assert isfile(join(out_path, 'denoised_func_data_nonaggr.nii.gz'))
    assert isfile(join(out_path, 'feature_scores.txt'))
    assert isfile(join(out_path, 'mask.nii.gz'))
    assert isfile(join(out_path, 'melodic_IC_thr.nii.gz'))
    assert isfile(join(out_path, 'melodic_IC_thr_MNI2mm.nii.gz'))

    # Check classification overview file
    classification_overview = pd.read_csv(join(out_path, 'classification_overview.txt'), sep='\t', index_col='IC')
    overview_true = np.array([True, 0.66, 0.65, 0.96, 0.0], dtype=object)
    assert classification_overview.loc[1].values[0]
    assert np.allclose(classification_overview.loc[1].values[1:].astype(float), overview_true[1:].astype(float), atol=0.3)

    # Check feature scores
    f_scores = np.loadtxt(join(out_path, 'feature_scores.txt'))
    f_true = np.array([6.563544605388391684e-01, 6.510340668773902939e-01, 9.635568513119533440e-01, 4.486414893783927278e-03])
    assert np.allclose(f_scores[0], f_true, atol=0.01)

    # Check motion ICs
    mot_ics = np.loadtxt(join(out_path, 'classified_motion_ICs.txt'), delimiter=',')
    assert (np.in1d(np.array([1, 2, 3, 4]), mot_ics.astype(int))).all()
