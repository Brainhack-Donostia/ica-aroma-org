# ica-aroma-org

[![codecov](https://codecov.io/gh/Brainhack-Donostia/ica-aroma-org/branch/master/graph/badge.svg)](https://codecov.io/gh/Brainhack-Donostia/ica-aroma-org)
[![CircleCI](https://circleci.com/gh/Brainhack-Donostia/ica-aroma-org.svg?branch=master&style=shield)](https://circleci.com/gh/Brainhack-Donostia/ica-aroma-org)

Repository to prepare the ICA-AROMA tutorial for Brainhack Donostia.

## License info

This repository is a clone of [maartenmennes/ICA-AROMA](https://github.com/maartenmennes/ICA-AROMA),
which has an Apache 2.0 license. We plan to modify this clone in order to restructure it into a
pure Python package, but we will make sure to fulfill the requirements of the license.

#### Source of the Gaussian-Gamma Mixture Model

The Gaussian-Gamma Mixture Model used in this package (`aroma.mixture.GGM`) was taken from
the [`nipy`](https://github.com/nipy/nipy) package. As such, the file in which the model
is defined (`aroma/mixture.py`) falls under `nipy`'s licensing. The license for that file
is stored in the file's docstring.
