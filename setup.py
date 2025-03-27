from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import os
import subprocess
import sys
import pybind11


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

        # Initialize cmake args with the standard options
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
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

        # Debug output
        print(f"Building with compilers: CC={cc}, CXX={cxx}")
        print(f"Building with Python: {python_executable}")
        if python_include_dir:
            print(f"Python include dir: {python_include_dir}")
        if python_library:
            print(f"Python library: {python_library}")
        print(f"CMake arguments: {cmake_args}")

        if "CXXFLAGS" not in os.environ:
            os.environ["CXXFLAGS"] = "-D_GLIBCXX_USE_CXX11_ABI=0"

        build_args = ['--config', 'Release']
        os.makedirs(self.build_temp, exist_ok=True)

        try:
            # Pass environment variables to CMAKE
            env = os.environ.copy()
            subprocess.check_call(['cmake', os.path.abspath('.')] + cmake_args, cwd=self.build_temp, env=env)
            subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp, env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error during CMake configuration or build: {e}")
            sys.exit(1)

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
