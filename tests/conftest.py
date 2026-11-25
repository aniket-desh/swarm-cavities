def pytest_configure(config):
    config.addinivalue_line("markers", "slow: large-M Monte Carlo or large-N simulation (minutes); deselect with -m 'not slow'")
