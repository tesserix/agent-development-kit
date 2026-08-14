"""What a consumer gets from the installed distribution, as opposed to from this tree.

The version comes from the distribution rather than a literal, because a literal is a
second place it can be wrong and the tag is the only one that decides. And nothing on an
import path a consumer walks may reach for a dependency they were never given.
"""

import os
import subprocess
import sys
from importlib.metadata import version

import pytest
from tools.versions import parts

from tesserix_adk import __version__, testing


def _imports(statement: str) -> subprocess.CompletedProcess[str]:
    """Run an import in a fresh interpreter, since this one already has the test runner.

    Coverage's subprocess hook imports pytest itself, so it is kept out of the child:
    otherwise the measurement decides the result of the very thing being measured.
    """
    clean = {key: value for key, value in os.environ.items() if not key.startswith("COV_CORE_")}
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", statement],
        capture_output=True,
        text=True,
        check=False,
        env=clean,
    )


def test_the_version_is_the_installed_distribution_version() -> None:
    assert __version__ == version("tesserix-adk")


def test_the_version_names_a_release() -> None:
    """A development build still says which release it comes before."""
    assert parts(__version__) >= (0, 0, 1)


def test_reaching_for_a_fake_does_not_import_the_test_runner() -> None:
    """`pytest` is a test-time dependency nobody installing the wheel is given, and
    importing it from a package `__init__` makes every fake unreachable without it."""
    done = _imports("import sys, tesserix_adk.testing; assert 'pytest' not in sys.modules")
    assert done.returncode == 0, done.stderr


def test_the_conformance_suites_are_still_importable_by_name() -> None:
    """Deferring the import must not hide the suites from the consumers that run them."""
    done = _imports("from tesserix_adk.testing import TracerConformance")
    assert done.returncode == 0, done.stderr


def test_a_name_the_package_does_not_have_still_fails_as_an_attribute_error() -> None:
    """An import hook that swallows typos turns them into a failure much further away."""
    with pytest.raises(AttributeError, match="NotAThing"):
        testing.__getattr__("NotAThing")
