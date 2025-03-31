sudo rm -rf build
sudo rm -rf temporal_random_walk.egg-info/
sudo rm -rf dist
sudo rm -rf wheelhouse

docker build -t temporal-walk-builder -f build_scripts/Dockerfile .

docker run --gpus all -v $(pwd):/project temporal-walk-builder
