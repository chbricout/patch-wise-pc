cp ./pyproject.toml clusters/docker/pyproject.toml
cd clusters/docker

docker build -t chbricout/patch-wise-pc .
docker push chbricout/patch-wise-pc 

rm pyproject.toml