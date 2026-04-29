#!/bin/bash

IMAGE_NAME="kimm-doosan"

echo "Allowing local user to access the X Window display..."
xhost +local:root

echo "Starting the ROS 2 container with GUI enabled..."

# Run the docker container and automatically execute the emulator script
docker run -it --rm \
    --net host \
    --privileged \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/dev:/dev:rw" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="/var/run/docker.sock:/var/run/docker.sock:rw" \
    --gpus all \
    $IMAGE_NAME \
    bash -c "echo 'Installing/Starting Doosan Emulator...' && \
             cd /ros2_ws/src/doosan-robot2 && \
             chmod +x ./install_emulator.sh && \
             ./install_emulator.sh && \
             echo 'Emulator ready! Dropping to terminal...' && \
             exec bash"

echo "Container closed. Revoking X Window display access..."
xhost -local:root
