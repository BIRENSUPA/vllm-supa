# Copyright©2020-2026 Shanghai Biren Technology Co., Ltd. All rights reserved.

import os
import sys
import shutil
import stat
import subprocess
from importlib.metadata import version
from setuptools import find_namespace_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from distutils.command.clean import clean

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)

build_metadata = os.getenv("BUILD_METADATA", default="no_build_metedata")

semantic_version = None
expect_torch_ver = None
# find correct version
with open(get_path("upstream_version.txt")) as fi:
    for line in fi.readlines():
        if line.startswith("VLLM_VERSION"):
            semantic_version = line.split('=')[1].strip()
        elif line.startswith("PYTORCH_VERSION"):
            expect_torch_ver = line.split('=')[1].strip()

assert semantic_version and expect_torch_ver, "can't query version."
if len(semantic_version) == 40:
    semantic_version = version("vllm")

torch_ver = subprocess.check_output("python3 -m pip list| grep 'torch '", shell=True).decode()
assert expect_torch_ver in torch_ver, f"Wrong pytorch version: {torch_ver}, expect {expect_torch_ver}"

def install_hooks():
    if os.path.isfile(".git"):
        # no need to install hook for worktree.
        return True
    source_hook = os.path.join(".env_setup", "pre-commit")
    git_hooks_dir = os.path.join(".git", "hooks")
    target_hook = os.path.join(git_hooks_dir, "pre-commit")
    if not os.path.exists(git_hooks_dir):
        print(f"Error: Directory not found {git_hooks_dir}")
        print("Please ensure you're running this command from the Git repo root")
        return False

    if os.path.exists(target_hook):
        with open(source_hook, "r") as src, open(target_hook, "r") as dst:
            if src.read() == dst.read():
                return True

    try:
        shutil.copy2(source_hook, target_hook)
        print(f"Hook file copied: {source_hook} -> {target_hook}")

        st = os.stat(target_hook)
        os.chmod(target_hook, st.st_mode | stat.S_IEXEC)
        print(f"Execute permission added: {target_hook}")
        return True
    except Exception as e:
        print(f"Installation failed: {str(e)}")
        return False


if install_hooks():
    print("Hook installed successfully! Commit messages will be validated on next commit")
else:
    print("Hook installation failed, please check errors")
    sys.exit(1)


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        with open(get_path("README.md"), encoding="utf-8") as f:
            return f.read()
    else:
        return ""


def get_full_version() -> str:
    local_version_separator = "." if "+" in semantic_version else "+"
    return f"{semantic_version}{local_version_separator}br2xx"


def generate_version_file() -> None:
    with open(get_path("vllm_supa", "version.py"), "w") as fo:
        fo.write(f"vllm='{semantic_version}'\n")
        fo.write(f"vllm_supa='{get_full_version()}'\n")
        meta = os.getenv("BUILD_METADATA")
        if meta:
            _, _, hash, timestamps = meta.split(".")
            fo.write(f"vllm_supa_hash='{hash}'\n")
            fo.write(f"built_time='{timestamps}'\n")


def get_requirements() -> list:
    """Get Python package dependencies from requirements.txt."""

    def _read_requirements(filename: str) -> list:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            line = line.strip()
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif line and not line.startswith("--") and not line.startswith("#"):
                resolved_requirements.append(line)
        return resolved_requirements

    return _read_requirements("requirements.txt")


def is_ninja_available() -> bool:
    return shutil.which("ninja") is not None


def build_extension():
    commands = list(arg.lower() for arg in sys.argv[1:])
    is_develop_mode = any(cmd in commands for cmd in ["develop", "editable", "editable_wheel"])

    debug = os.getenv("DEBUG", "0").lower() in ("1", "on", "yes", "y")
    enable_coverage = os.getenv("ENABLE_COVERAGE", "OFF").lower() in ("1", "on", "yes", "y")

    build_type = "Debug" if debug or enable_coverage or is_develop_mode else "Release"

    build_temp = "build"

    num_jobs = int(os.getenv("MAX_JOBS", 16))
    src_folder = os.path.abspath(".")

    # Create build directory if it does not exist.
    if not os.path.exists(build_temp):
        os.makedirs(build_temp)

    # Keep the CMake install target aligned with the editable package.  This
    # also makes setup.py safe to run after a direct CMake build that used a
    # stale or system-wide install prefix from the existing cache.
    cmake_args = [
        f"-DCMAKE_BUILD_TYPE={build_type}",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        f"-DCMAKE_INSTALL_PREFIX={src_folder}",
    ]

    if enable_coverage:
        cmake_args.append("-DENABLE_COVERAGE=ON")

    # Set USE_MARLIN_KERNELS=OFF to skip the slow Marlin GEMM kernels.
    if os.getenv("USE_MARLIN_KERNELS", "ON").lower() in ("0", "off", "no", "n"):
        cmake_args.append("-DUSE_MARLIN_KERNELS=OFF")

    if is_ninja_available():
        build_tool = ["-G", "Ninja"]
        cmake_args += [
            "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
            "-DCMAKE_JOB_POOLS:STRING=compile={}".format(num_jobs),
        ]
    else:
        # Default build tool to whatever cmake picks.
        build_tool = []

    subprocess.check_call(["cmake", src_folder, *build_tool, *cmake_args], cwd=build_temp)

    build_args = ["--build", ".", f"-j={num_jobs}"]

    env = os.environ.copy()
    env["USE_CCACHE"] = "1" if shutil.which("ccache") else "0"
    subprocess.check_call(["cmake", *build_args], cwd=build_temp, env=env)

    target = get_path("vllm_supa")

    for file in filter(lambda x: x.endswith(".so") or ".so." in x, os.listdir(build_temp)):
        src_path = os.path.join(build_temp, file)
        dst_path = os.path.join(target, file)
        shutil.copy(src_path, dst_path)
        print(f"Copy: {src_path} -> {dst_path}")


class custom_develop(develop):
    def run(self):
        build_extension()
        super().run()


class custom_build_info(build_py):
    def run(self):
        build_extension()
        super().run()


class Clean(clean):
    def run(self):
        shutil.rmtree("build", ignore_errors=True)
        shutil.rmtree("vllm_supa.egg-info", ignore_errors=True)
        for filename in os.listdir("vllm_supa"):
            if filename.endswith(".so") or ".so." in filename:
                os.remove("vllm_supa/" + filename)


generate_version_file()

setup(
    name="vllm_supa",
    version=get_full_version(),
    author="Shanghai Biren Technology Co., Ltd",
    description=("vLLM Biren backend plugin " + " --> build_metadata:{}".format(build_metadata)),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/vllm-project/vllm",
    project_urls={
        "Homepage": "https://github.com/vllm-project/vllm",
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    packages=find_namespace_packages(include=("vllm_supa*",)),
    package_data={
        "vllm_supa": ["*.so", "*.so.*"],
    },
    python_requires=">=3.10",
    install_requires=get_requirements(),
    cmdclass={"build_py": custom_build_info, "develop": custom_develop, "clean": Clean},
    entry_points={
        "vllm.platform_plugins": ["biren = vllm_supa:register"],
        "vllm.general_plugins": [
            "biren_enhanced_model = vllm_supa:register_model",
            "biren_patch = vllm_supa:register_patch",
        ],
    },
)
