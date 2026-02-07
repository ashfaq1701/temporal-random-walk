# Clean
rm -rf build dist *.egg-info

# Make sure toolchain is consistent (optional but helpful)
which gcc
gcc --version
which ld
ld --version

# Force setup.py to use the same gcc/g++ you have in PATH (GCCcore 12.2)
export CC=$(which gcc)
export CXX=$(which g++)

# ASAN + debug
export DEBUG_BUILD=1
export CFLAGS="-fsanitize=address -g -O1 -fno-omit-frame-pointer"
export CXXFLAGS="-fsanitize=address -g -O1 -fno-omit-frame-pointer"
export LDFLAGS="-fsanitize=address"

python setup.py bdist_wheel
