import numpy as np
import os.path as op
import pandas as pd

import pytest

from aroma import features, utils

def test_feature_time_series(nilearn_data):

    # Get path to confounds file
    test_path, _ = op.split(nilearn_data.func[0])

    # Read confounds
    confounds = pd.read_csv(nilearn_data.confounds[0], sep='\t')

    # Extract motion parameters from confounds
    mc = confounds[["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]]
    mc_path = op.join(test_path, 'mc.tsv')
    mc.to_csv(mc_path, sep='\t', index=False, header=None)

    # Get path to melmix file
    cwd = utils.get_resource_path()
    melmix = op.join(cwd, 'aroma', 'resources', 'melodic_mix')

    # Run feature_time_series
    maxRPcorr = features.feature_time_series(melmix, mc_path)

    # Expected values
    true_maxRPcorr = np.array([0.65255575, 0.86003032, 0.88690363, 0.61399576, 0.43840624])

    assert np.allclose(maxRPcorr[:len(true_maxRPcorr)], true_maxRPcorr, atol=1e-2)


def test_feature_frequency(nilearn_data):

    # Get path to melmix file
    cwd = utils.get_resource_path()
    melTmix = op.join(cwd, 'aroma', 'resources', 'melodic_FTmix')

    HFC = features.feature_frequency(melTmix, TR=2)

    # Expected values
    true_HFC = np.array([0.96279762, 0.08234127, 0.13194444, 0.96279762, 0.04513889])

    assert np.allclose(HFC[:len(true_HFC)], true_HFC, atol=1e-2)
