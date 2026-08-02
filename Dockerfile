# Use ROS 2 Jazzy as the base image
FROM osrf/ros:jazzy-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

SHELL ["/bin/bash", "-c"]

# Install base dependencies and ROS 2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    tmux \
    wget \
    curl \
    gnupg \
    lsb-release \
    ca-certificates \
    docker.io \
    libpoco-dev \
    libyaml-cpp-dev \
    dbus-x11 \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    x11-apps \
    mesa-utils \
    vim \
    python3-pip \
    python3-venv \
    ros-jazzy-control-msgs \
    ros-jazzy-realtime-tools \
    ros-jazzy-xacro \
    ros-jazzy-joint-state-publisher-gui \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-gazebo-msgs \
    ros-jazzy-moveit-msgs \
    ros-jazzy-moveit-configs-utils \
    ros-jazzy-moveit-ros-move-group \
    ros-jazzy-example-interfaces \
    'ros-jazzy-librealsense2*' \
    ros-jazzy-realsense2-camera \
    ros-jazzy-realsense2-description \
    ros-jazzy-ur \
    && rm -rf /var/lib/apt/lists/*

# Set up the OSRF Gazebo package repository
RUN echo \
    "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable \
    $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/gazebo-stable.list \
    && wget -qO- http://packages.osrfoundation.org/gazebo.key \
        | apt-key add - \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ros-jazzy-gazebo-msgs \
        ros-jazzy-ros-gz \
        ros-jazzy-gz-ros2-control \
    && rm -rf /var/lib/apt/lists/*

# Create the ROS 2 workspace source directory
WORKDIR /ros2_ws/src

# GitHub token supplied at build time
ARG GIT_TOKEN

# Clone the Doosan repository
RUN git clone \
        "https://${GIT_TOKEN}@github.com/purduelamm/KIMM-doosan-robot2.git" \
        doosan-robot2 \
    && cd doosan-robot2 \
    && git checkout jazzy

# Clone the UR Gazebo repository and its submodules
RUN git clone \
        --recursive \
        "https://${GIT_TOKEN}@github.com/purduelamm/KIMM-UR-gazebo.git" \
        KIMM-UR-gazebo

# Move to the workspace root
WORKDIR /ros2_ws

# Install ROS package dependencies
RUN apt-get update \
    && rosdep update \
    && rosdep install \
        --from-paths src \
        --ignore-src \
        --rosdistro jazzy \
        --recursive \
        -y \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated Python environment.
# --system-site-packages preserves access to Ubuntu Python packages.
RUN python3 -m venv \
    --system-site-packages \
    /opt/ros_venv

# Use the virtual environment by default
ENV VIRTUAL_ENV=/opt/ros_venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Install mutually compatible Python packages
RUN python -m pip install --no-cache-dir \
    "numpy==1.26.4" \
    "scipy==1.14.1" \
    "open3d==0.19.0" \
    "trimesh>=4,<5" \
    "rtree>=1,<2"

# Verify rclpy after sourcing the ROS environment
RUN source /opt/ros/jazzy/setup.bash \
    && python - <<'PY'
import sys
import rclpy

print("Python executable:", sys.executable)
print("rclpy            :", rclpy.__file__)
print("ROS Python import: OK")
PY

# Build the ROS 2 workspace
RUN source /opt/ros/jazzy/setup.bash \
    && source /opt/ros_venv/bin/activate \
    && colcon build --symlink-install

# Automatically configure interactive shells
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc \
    && echo "source /opt/ros_venv/bin/activate" >> /root/.bashrc \
    && echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc

WORKDIR /ros2_ws

CMD ["bash"]
