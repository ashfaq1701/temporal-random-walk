rm -rf build
rm -rf dist
rm -rf wheelhouse

docker build -t temporal-walk-builder \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  -f build_scripts/Dockerfile .

docker run --gpus all -v $(pwd):/project temporal-walk-builder
