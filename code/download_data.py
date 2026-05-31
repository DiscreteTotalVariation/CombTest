#!/usr/bin/env python3
"""Download and extract precomputed data archives from Dropbox.

Downloads:
  - exact_distributions.zip  (exact DTV distributions)
  - cdf_exact.zip            (exact CDFs)
  - cvm_beta.zip             (fitted beta parameters)
  - cvm_gamma.zip            (fitted gamma parameters)
"""

import os
import sys
import zipfile
import urllib.request
from tqdm import tqdm

ARCHIVES = {
    "exact_distributions": (
        "https://www.dropbox.com/scl/fi/l9wpmgeuzc2onmwa326am/"
        "exact_distributions.zip?rlkey=xsvbspq9wdhr5hywjdmwco89x&st=pecim0fx&dl=1",
        None,  # extract to data_dir directly
    ),
    "cdf_exact": (
        "https://www.dropbox.com/scl/fi/7xx7kz06c1s7kj8agx71q/"
        "cdf_exact.zip?rlkey=kyk3z8n3u5n3vdx809wi91b0a&st=p8swnhuw&dl=1",
        None,
    ),
    "cvm_beta": (
        "https://www.dropbox.com/scl/fi/k4ok53ucdcvpnwpifwhi3/"
        "cvm_beta.zip?rlkey=kph6f55fk191e59eo7ak0g486&st=r8ckqtfv&dl=1",
        "fitted",  # extract into data_dir/fitted/
    ),
    "cvm_gamma": (
        "https://www.dropbox.com/scl/fi/xcpysto3nk4f5tdq233h1/"
        "cvm_gamma.zip?rlkey=nm4c87uqev4v2ivwk5flk3i0n&st=w75xq1j6&dl=1",
        "fitted",
    ),
}


def download_with_progress(url, dest_path, description="Downloading"):
    response = urllib.request.urlopen(url)
    total = int(response.headers.get("Content-Length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=description
    ) as pbar:
        while True:
            chunk = response.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            pbar.update(len(chunk))


def extract_with_progress(zip_path, dest_dir, description="Extracting"):
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc=description, unit=" files"):
            zf.extract(member, dest_dir)


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    for name, (url, subdir) in ARCHIVES.items():
        zip_path = os.path.join(data_dir, f"{name}.zip")
        extract_dir = os.path.join(data_dir, subdir) if subdir else data_dir
        os.makedirs(extract_dir, exist_ok=True)

        print(f"\n=== {name} ===")
        download_with_progress(url, zip_path, f"Downloading {name}.zip")
        extract_with_progress(zip_path, extract_dir, f"Extracting {name}.zip")
        os.remove(zip_path)
        print(f"Cleaned up {name}.zip")

    print("\nDone. Data ready in:")
    print("  ../data/exact_distributions/")
    print("  ../data/cdf_exact/")
    print("  ../data/fitted/cvm_beta/")
    print("  ../data/fitted/cvm_gamma/")


if __name__ == "__main__":
    main()
