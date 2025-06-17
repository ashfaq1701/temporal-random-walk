import os
import re
import subprocess
import sys
import pybind11
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

def find_cmake():
    try:
        subprocess.check_output(['cmake', '--version'])
    except subprocess.CalledProcessError:
        raise RuntimeError("CMake must be installed to build the following extensions: _temporal_random_walk")


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        find_cmake()
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Check for debug mode via environment variable
        is_debug = os.environ.get('DEBUG_BUILD', '0').lower() in ('1', 'true', 'yes')
        build_type = 'Debug' if is_debug else 'Release'

        print(f"Building in {build_type} mode")

        # Initialize cmake args with the standard options
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DCMAKE_BUILD_TYPE={build_type}',
        ]

        # Find compilers - Use environment variables or defaults
        cc = os.environ.get('CC', '/usr/bin/gcc')
        cxx = os.environ.get('CXX', '/usr/bin/g++')

        # Add compiler paths to cmake args
        cmake_args.append(f'-DCMAKE_C_COMPILER={cc}')
        cmake_args.append(f'-DCMAKE_CXX_COMPILER={cxx}')

        # Use environment-defined Python paths if available
        python_executable = os.environ.get('PYTHON_EXECUTABLE', sys.executable)
        python_include_dir = os.environ.get('PYTHON_INCLUDE_DIR', '')
        python_library = os.environ.get('PYTHON_LIBRARY', '')

        # Add Python paths to cmake args
        cmake_args.append(f'-DPYTHON_EXECUTABLE={python_executable}')

        if python_include_dir:
            cmake_args.append(f'-DPYTHON_INCLUDE_DIR={python_include_dir}')

        if python_library:
            cmake_args.append(f'-DPYTHON_LIBRARY={python_library}')

        # Add pybind11 include directory
        cmake_args.append(f'-DPYBIND11_INCLUDE_DIR={pybind11.get_include()}')

        # Detect CUDA version
        cuda_version = self.detect_cuda_version()

        print(f"Using CUDA Version: {cuda_version}")
        cmake_args.append(f'-DCUDA_VERSION={cuda_version}')

        # Set up compilation flags based on build type
        if is_debug:
            # Debug flags for C++
            base_cxxflags = "-D_GLIBCXX_USE_CXX11_ABI=0 -g -O0 -fno-omit-frame-pointer"
            # CUDA debug flags
            base_cudaflags = "-g -G -O0"
            print("Using debug compilation flags")
        else:
            # Release flags
            base_cxxflags = "-D_GLIBCXX_USE_CXX11_ABI=0"
            base_cudaflags = ""
            print("Using release compilation flags")

        # Merge with existing flags if any
        if "CXXFLAGS" in os.environ:
            os.environ["CXXFLAGS"] += " " + base_cxxflags
        else:
            os.environ["CXXFLAGS"] = base_cxxflags

        if "CUDAFLAGS" in os.environ:
            os.environ["CUDAFLAGS"] += " " + base_cudaflags
        else:
            os.environ["CUDAFLAGS"] = base_cudaflags

        # NVCC flags need to be passed through CMake
        if is_debug:
            cmake_args.append('-DCMAKE_CUDA_FLAGS=-g -G -O0')

        build_args = ['--config', build_type]
        os.makedirs(self.build_temp, exist_ok=True)

        try:
            # Pass environment variables to CMAKE
            env = os.environ.copy()
            print(f"Environment variables during build:")
            print(f"  CXXFLAGS: {env.get('CXXFLAGS', 'not set')}")
            print(f"  CUDAFLAGS: {env.get('CUDAFLAGS', 'not set')}")

            subprocess.check_call(['cmake', os.path.abspath('.')] + cmake_args, cwd=self.build_temp, env=env)
            subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp, env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error during CMake configuration or build: {e}")
            sys.exit(1)

    def detect_cuda_version(self):
        """
        Detect CUDA version using `nvcc` or `cuda-python`.
        Returns a string representing the CUDA version (e.g., '10.2', '11.0').
        """
        # Try detecting the CUDA version from the environment
        try:
            cuda_version = subprocess.check_output(['nvcc', '--version']).decode()
            match = re.search(r'V(\d+\.\d+)', cuda_version)
            if match:
                return match.group(1)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Fallback to using CUDA-Python package if available
        try:
            import cuda
            return cuda.__version__
        except ImportError:
            pass

        return 'none'


def read_version_number():
    with open('version_number.txt', 'r') as file:
        version_number = file.readline()
    return version_number.strip()


setup(
    name="temporal_random_walk",
    version=read_version_number(),
    author="Ashfaq Salehin",
    author_email="ashfaq.salehin1701@gmail.com",
    description="A library to sample temporal random walks from in-memory temporal graphs",
    long_description=open('README.md').read(),
    packages=find_packages(),
    package_data={
        'temporal_random_walk': ['*.so'],
    },
    include_package_data=True,
    long_description_content_type="text/markdown",
    url="https://github.com/ashfaq1701/temporal-random-walk",
    ext_modules=[CMakeExtension('_temporal_random_walk')],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=["pybind11>=2.6.0", "numpy", "networkx"],
)
