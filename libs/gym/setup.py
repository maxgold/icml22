import os.path
import sys
import itertools

from setuptools import find_packages, setup

# Don't import gym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gym"))
from version import VERSION

# Environment-specific dependencies.
extras = {
    "atari": ["ale-py~=0.7.1"],
    "accept-rom-license": ["autorom[accept-rom-license]~=0.4.2"],
    "box2d": ["box2d-py==2.3.5", "pyglet>=1.4.0"],
    "classic_control": ["pyglet>=1.4.0"],
    "mujoco": ["mujoco_py>=1.50, <2.0"],
    "toy_text": ["scipy>=1.4.1"],
    "other": ["lz4>=3.1.0", "opencv-python>=3.0"],
}

# Meta dependency groups.
nomujoco_blacklist = set(["mujoco", "accept-rom-license", "atari"])
nomujoco_groups = set(extras.keys()) - nomujoco_blacklist

extras["nomujoco"] = list(
    itertools.chain.from_iterable(map(lambda group: extras[group], nomujoco_groups))
)


all_blacklist = set(["accept-rom-license"])
all_groups = set(extras.keys()) - all_blacklist

extras["all"] = list(
    itertools.chain.from_iterable(map(lambda group: extras[group], all_groups))
)

setup(
    name="gym",
    version=VERSION,
    description="Gym: A universal API for reinforcement learning environments.",
    url="https://github.com/openai/gym",
    author="Gym Community",
    author_email="jkterry@umd.edu",
    license="",
    packages=[package for package in find_packages() if package.startswith("gym")],
    zip_safe=False,
    extras_require=extras,
    package_data={
        "gym": [
            "envs/mujoco/assets/*.xml",
            "envs/classic_control/assets/*.png",
        ]
    },
    tests_require=["pytest", "mock"],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)