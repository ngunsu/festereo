docker rm -f $(docker ps -a -q)
docker build --tag=stereodock ./docker/
docker run -it --ipc=host --gpus all -v $PWD:/workspace -v $HOME/datasets:/root/datasets --env COMET_KEY=$COMET_KEY --env COMET_WORKSPACE=$COMET_WORKSPACE --env COMET_REST_KEY=$COMET_REST_KEY --env COMET_DISABLE_AUTO_LOGGING=$COMET_DISABLE_AUTO_LOGGING --name stereodock stereodock 
