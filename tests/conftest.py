# Copyright (c) 2019, Michael Boyle
# See LICENSE file for details: <https://github.com/moble/spherical_functions/blob/master/LICENSE>

import os
import pytest
import numpy as np
import quaternion
from spherical_functions import ell_max as ell_max_default


def pytest_addoption(parser):
    parser.addoption("--ell_max", action="store", type=int, default=ell_max_default,
                     help="Maximum ell value to test")
    parser.addoption("--run_slow_tests", action="store_true", default=False,
                     help="Run all tests, including slow ones")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run_slow_tests"):
        return
    skip_slow = pytest.mark.skip(reason="need --run_slow_tests option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_runtest_setup(item):
    if 'slow' in item.keywords and not item.config.getoption("--run_slow_tests"):
        pytest.skip("Need `--run_slow_tests` command-line argument to run")


@pytest.fixture
def ell_max(request):
    return request.config.getoption("--ell_max")


@pytest.fixture
def special_angles():
    return np.arange(-1 * np.pi, 1 * np.pi + 0.1, np.pi / 4.)


@pytest.fixture
def Rs():
    ones = [0, -1., 1.]
    rs = [np.quaternion(w, x, y, z).normalized() for w in ones for x in ones for y in ones for z in ones][1:]
    np.random.seed(1842)
    rs = rs + [r.normalized() for r in [np.quaternion(np.random.uniform(-1, 1), np.random.uniform(-1, 1),
                                                      np.random.uniform(-1, 1), np.random.uniform(-1, 1))
                                        for i in range(20)]]
    return np.array(rs)

