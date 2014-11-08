import pytest

def pytest_addoption(parser):
    parser.addoption("--ell_max", action="store", type=int, default=4,
                     help="Maximum ell value to test")

@pytest.fixture
def ell_max(request):
    return request.config.getoption("--ell_max")
