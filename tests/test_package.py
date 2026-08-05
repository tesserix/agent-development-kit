from tesserix_adk import __version__


def test_version_is_exported() -> None:
    assert __version__ == "0.0.1"
