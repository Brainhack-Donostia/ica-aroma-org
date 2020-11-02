import numpy as np
import pandas as pd
import subprocess
from os.path import join, split

import pytest


def test_integration(skip_integration, nilearn_data):
    if skip_integration:
        pytest.skip('Skipping integration test')

    test_path, _ = split(nilearn_data.func[0])

    subprocess.run(f'aroma -o {test_path} -i {nilearn_data.func[0]} -mc {nilearn_data.confounds[0]}')
