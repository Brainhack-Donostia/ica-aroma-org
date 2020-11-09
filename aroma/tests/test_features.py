import numpy as np
import os.path as op
import pandas as pd

from aroma import features
from aroma.tests.utils import get_tests_resource_path


def test_feature_time_series(nilearn_data):

    # Get path to confounds file
    temp_path, _ = op.split(nilearn_data.func[0])

    # Read confounds
    confounds = pd.read_csv(nilearn_data.confounds[0], sep="\t")

    # Extract motion parameters from confounds
    mc = confounds[["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]]
    mc_file = op.join(temp_path, "mc.tsv")
    mc.to_csv(mc_file, sep="\t", index=False, header=None)

    # Get path to mel_mix file
    test_path = op.join(get_tests_resource_path(), "integration_test_ground_truth")
    mel_mix = op.join(test_path, "melodic_mix")

    # Run feature_time_series
    max_RP_corr = features.feature_time_series(mel_mix, mc_file)

    # Expected values
    true_features = op.join(test_path, "feature_scores.txt")
    true_features = np.loadtxt(true_features)
    true_max_RP_corr = true_features[:, 0]

    assert np.allclose(max_RP_corr, true_max_RP_corr, atol=1e-2)


def test_feature_frequency(nilearn_data):

    # Get path to mel_mix file
    test_path = op.join(get_tests_resource_path(), "integration_test_ground_truth")
    mel_T_mix = op.join(test_path, "melodic_FTmix")

    HFC = features.feature_frequency(mel_T_mix, TR=2)

    # Expected values
    true_features = op.join(test_path, "feature_scores.txt")
    true_features = np.loadtxt(true_features)
    true_HFC = true_features[:, 2]

    assert np.allclose(HFC, true_HFC, atol=1e-2)
