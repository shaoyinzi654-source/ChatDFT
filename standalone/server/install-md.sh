#!/usr/bin/env bash
# Ubuntu 22.04, using ChatDFT's existing Python environment. No credentials stored.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
task_root=/opt/chatdft/md
mkdir -p "$task_root/tools"
apt-get update
apt-get install -y --no-install-recommends gromacs cmake build-essential libfftw3-dev ca-certificates curl
/opt/chatdft/venv/bin/python -m pip install 'mdtraj>=1.10,<2' 'scipy>=1.13,<2'
cd "$task_root/tools"
archive=gromacs-2018.8.tar.gz
if [ ! -f "$archive" ]; then
    curl --fail --location --retry 2 --connect-timeout 20 --max-time 600 \
        https://ftp.gromacs.org/gromacs/gromacs-2018.8.tar.gz -o "$archive.download"
    mv "$archive.download" "$archive"
fi
echo "776923415df4bc78869d7f387c834141fdcda930b2e75be979dc59ecfa6ebecf  $archive" | sha256sum --check -
if [ ! -x "$task_root/gromacs-2018/bin/gmx" ]; then
    tar xzf "$archive"
    # GROMACS 2018 predates GCC 11's stricter standard header dependencies.
    for source in gromacs-2018.8/src/gromacs/awh/biasparams.cpp gromacs-2018.8/src/gromacs/mdlib/minimize.cpp; do
        if ! grep -q '^#include <limits>' "$source"; then
            sed -i '1i#include <limits>' "$source"
        fi
    done
    cmake -S gromacs-2018.8 -B build-2018 \
        -DCMAKE_INSTALL_PREFIX="$task_root/gromacs-2018" \
        -DGMX_BUILD_OWN_FFTW=OFF -DGMX_GPU=OFF -DGMX_MPI=OFF \
        -DGMX_OPENMP=ON -DBUILD_TESTING=OFF -DGMX_BUILD_UNITTESTS=OFF
    cmake --build build-2018 -j 8
    cmake --install build-2018
fi
/usr/bin/gmx --version
"$task_root/gromacs-2018/bin/gmx" --version
/opt/chatdft/venv/bin/python -c 'import mdtraj, scipy; print("ChatDFT MD dependencies ready")'
