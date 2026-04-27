# 1. Use ROS 2 Jazzy as the base image
FROM osrf/ros:jazzy-desktop-full 

# 2. Install base dependencies and ROS 2 packages
RUN apt-get update && apt-get install -y \
    git \
    tmux \
    wget \
    curl \
    gnupg \
    lsb-release \
    docker.io \
    libpoco-dev \
    libyaml-cpp-dev \
    ros-jazzy-control-msgs \
    ros-jazzy-realtime-tools \
    ros-jazzy-xacro \
    ros-jazzy-joint-state-publisher-gui \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-gazebo-msgs \
    ros-jazzy-moveit-msgs \
    dbus-x11 \
    ros-jazzy-moveit-configs-utils \
    ros-jazzy-moveit-ros-move-group \
    ros-jazzy-example-interfaces \
    vim \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 3. Setup Gazebo packages
RUN sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list' \
    && wget http://packages.osrfoundation.org/gazebo.key -O - | apt-key add - \
    && apt-get update \
    && apt-get install -y ros-jazzy-gazebo-msgs ros-jazzy-ros-gz ros-jazzy-gz-ros2-control \
    && rm -rf /var/lib/apt/lists/*

# 4. Create the ROS 2 workspace directory
WORKDIR /ros2_ws/src

# 5. Define the token as a build argument
ARG GIT_TOKEN

# 6. Clone the private repository
RUN git clone https://${GIT_TOKEN}@github.com/purduelamm/KIMM-doosan-robot2.git doosan-robot2 \
    && cd doosan-robot2 \
    && git checkout jazzy \
    && apt-get update \
    && rosdep install -r --from-paths . --ignore-src --rosdistro $ROS_DISTRO -y \
    && rm -rf /var/lib/apt/lists/*

# 7. Set the working directory to the root of the workspace
WORKDIR /ros2_ws

# 8. Install remaining dependencies using rosdep (catch-all)
RUN apt-get update && rosdep install --from-paths src --ignore-src -r -y \
    && rm -rf /var/lib/apt/lists/*

# 9. Build the workspace
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && colcon build"

# 10. Set the entrypoint to automatically source ROS 2 and your workspace
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
RUN echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc
CMD ["bash"]