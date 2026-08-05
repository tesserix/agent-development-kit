"""The version a consumer sees comes from the installed distribution, not a literal.

A literal in the source is a second place the version can be wrong; the tag is the only
one that decides. See docs/releasing.md.
"""

from importlib.metadata import version

from tools.versions import parts

from tesserix_adk import __version__


def test_the_version_is_the_installed_distribution_version() -> None:
    assert __version__ == version("tesserix-adk")


def test_the_version_names_a_release() -> None:
    """A development build still says which release it comes before."""
    assert parts(__version__) >= (0, 0, 1)
