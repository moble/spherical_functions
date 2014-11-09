import pytest
from spherical_functions import ell_max as ell_max_default

def pytest_addoption(parser):
    parser.addoption("--ell_max", action="store", type=int, default=ell_max_default,
                     help="Maximum ell value to test")

@pytest.fixture
def ell_max(request):
    return request.config.getoption("--ell_max")
