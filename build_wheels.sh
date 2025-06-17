sudo rm -rf build
sudo rm -rf temporal_random_walk.egg-info/
sudo rm -rf dist
sudo rm -rf wheelhouse

docker build -t temporal-walk-builder-12-6 -f build_scripts/build_12_6/Dockerfile .
docker run --gpus all -v $(pwd):/project temporal-walk-builder-12-6

docker build -t temporal-walk-builder-12-8 -f build_scripts/build_12_8/Dockerfile .
docker run --gpus all -v $(pwd):/project temporal-walk-builder-12-8
