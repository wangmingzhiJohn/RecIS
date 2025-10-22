import os
import re

import torch


def get_package_version():
    pwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(pwd, "recis", "__init__.py")) as f:
        groups = re.findall(r"__version__.*([0-9]+)\.([0-9]+)\.([0-9]+)", f.read())
        main_version, minor_version, patch_version = groups[0]
        print(f"RecIS version {main_version}.{minor_version}.{patch_version}")
        return main_version, minor_version, patch_version


def get_cuda_version():
    return torch.version.cuda


def get_version():
    version = get_package_version()
    torch_version_clean = torch.__version__.split(".git")[0]
    torch_version = f"torch{torch_version_clean.replace('.', '').replace('+', '')}"

    if torch.version.cuda is not None:
        cuda_version = f"cuda{torch.version.cuda.replace('.', '')}"
    elif torch.version.hip is not None:
        # hip version is messy
        cuda_version = ""
    else:
        raise RuntimeError(
            "Neither CUDA nor ROCm/HIP version found in PyTorch installation"
        )

    version = f"{'.'.join(version)}+{cuda_version}{torch_version}"
    return version


if __name__ == "__main__":
    print(get_version())
