from gammaboard import *


def test_open_dashboard():
    process = open_dashboard()
    process.terminate()
