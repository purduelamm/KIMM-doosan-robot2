# 
#  dsr_bringup2
#  Author: Minsoo Song (minsoo.song@doosan.com)
#  
#  Copyright (c) 2025 Doosan Robotics
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# 

import math
import os

from launch import LaunchDescription
from launch.actions import RegisterEventHandler,DeclareLaunchArgument, LogInfo
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition, UnlessCondition

from launch_ros.actions import Node, SetRemap
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription, SetLaunchConfiguration, GroupAction

from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import OpaqueFunction
from launch.launch_context import LaunchContext
from moveit_configs_utils import MoveItConfigsBuilder
from dsr_bringup2.utils import read_update_rate, show_git_info


EE_BOX_SIZE = (0.50, 0.50, 0.80)
ARUCO_MARKER_SIZE = 0.115
ARUCO_MARKER_THICKNESS = 0.001
# m0609 link_6 position in base coordinates at
# joints [90, -30, -35, 0, -115, 180] degrees.
EE_POSITION_IN_BASE = (-0.0062, -0.539021265629, 0.490459961276)
EE_BOX_SDF = f'''<?xml version="1.0"?>
<sdf version="1.8">
  <model name="ee_support_box">
    <static>true</static>
    <link name="box_link">
      <collision name="box_collision">
        <geometry>
          <box><size>{EE_BOX_SIZE[0]} {EE_BOX_SIZE[1]} {EE_BOX_SIZE[2]}</size></box>
        </geometry>
      </collision>
      <visual name="box_visual">
        <geometry>
          <box><size>{EE_BOX_SIZE[0]} {EE_BOX_SIZE[1]} {EE_BOX_SIZE[2]}</size></box>
        </geometry>
        <material>
          <ambient>0.18 0.32 0.50 1.0</ambient>
          <diffuse>0.25 0.45 0.70 1.0</diffuse>
          <specular>0.10 0.10 0.10 1.0</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''


def get_target_ee_world_position(context):
    """Return the target m0609 flange position in Gazebo world coordinates."""
    robot_position = [
        float(LaunchConfiguration(axis).perform(context)) for axis in ('x', 'y', 'z')
    ]
    roll, pitch, yaw = [
        float(LaunchConfiguration(axis).perform(context)) for axis in ('R', 'P', 'Y')
    ]

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    robot_rotation = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    return [
        robot_position[row]
        + sum(robot_rotation[row][column] * EE_POSITION_IN_BASE[column]
              for column in range(3))
        for row in range(3)
    ]


def spawn_box_below_end_effector(context):
    """Spawn a grounded 50x50x80 cm box below the target flange in XY."""
    ee_world = get_target_ee_world_position(context)
    box_center = (
        ee_world[0],
        ee_world[1],
        EE_BOX_SIZE[2] / 2.0,
    )

    return [
        Node(
            package='ros_gz_sim',
            executable='create',
            output='screen',
            arguments=[
                '-string', EE_BOX_SDF,
                '-name', 'ee_support_box',
                '-allow_renaming', 'false',
                '-x', str(box_center[0]),
                '-y', str(box_center[1]),
                '-z', str(box_center[2]),
            ],
            condition=IfCondition(LaunchConfiguration('gz')),
        )
    ]


def spawn_aruco_markers(context):
    """Create one Gazebo spawn action for each configured ArUco marker."""
    marker_specs = LaunchConfiguration('aruco_markers').perform(context).strip()
    if not marker_specs:
        return []

    if marker_specs.lower() == 'box_corners':
        box_x, box_y, _ = get_target_ee_world_position(context)
        marker_offset = (EE_BOX_SIZE[0] - ARUCO_MARKER_SIZE) / 2.0
        marker_z = EE_BOX_SIZE[2] + ARUCO_MARKER_THICKNESS / 2.0
        # IDs 4->1 and 3->2 form orthogonal diagonals, matching the pose estimator.
        parsed_marker_specs = [
            ('1', box_x + marker_offset, box_y + marker_offset, marker_z, 0, 0, 0),
            ('2', box_x - marker_offset, box_y + marker_offset, marker_z, 0, 0, 0),
            ('3', box_x + marker_offset, box_y - marker_offset, marker_z, 0, 0, 0),
            ('4', box_x - marker_offset, box_y - marker_offset, marker_z, 0, 0, 0),
        ]
    else:
        parsed_marker_specs = []
        for marker_spec in marker_specs.split(';'):
            fields = [field.strip() for field in marker_spec.split(',')]
            if len(fields) != 7:
                raise RuntimeError(
                    "Each aruco_markers entry must be 'id,x,y,z,roll,pitch,yaw'; "
                    f"received: {marker_spec!r}"
                )
            parsed_marker_specs.append(fields)

    marker_model_dir = os.path.join(
        get_package_share_directory('dsr_visualservoing'),
        'description',
    )
    spawn_actions = []

    for index, fields in enumerate(parsed_marker_specs, start=1):
        marker_id, x, y, z, roll, pitch, yaw = fields
        if marker_id not in {'1', '2', '3', '4'}:
            raise RuntimeError(
                f"Unsupported ArUco marker ID {marker_id!r}; available models are 1, 2, 3, and 4"
            )

        spawn_actions.append(
            Node(
                package='ros_gz_sim',
                executable='create',
                output='screen',
                arguments=[
                    '-file', os.path.join(marker_model_dir, f'markerbox{marker_id}.sdf'),
                    '-name', f'aruco_marker_{marker_id}_{index}',
                    '-x', str(x),
                    '-y', str(y),
                    '-z', str(z),
                    '-R', str(roll),
                    '-P', str(pitch),
                    '-Y', str(yaw),
                ],
                condition=IfCondition(LaunchConfiguration('gz')),
            )
        )

    return spawn_actions

def print_launch_configuration_value(context, *args, **kwargs):
    gz_value = LaunchConfiguration('gz').perform(context)
    print(f'LaunchConfiguration gz: {gz_value}')
    return gz_value

def move_group_fn(context):
    model_value = LaunchConfiguration('model').perform(context)
    package_name = f"dsr_moveit_config_{model_value}"

    moveit_config = (
        MoveItConfigsBuilder(model_value, "robot_description", package_name)
        .robot_description(file_path=f"config/{model_value}.urdf.xacro")
        .robot_description_semantic(file_path="config/dsr.srdf.xacro", mappings={'gripper': LaunchConfiguration('gripper')})
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(
            pipelines=["ompl", "chomp"],
            default_planning_pipeline="ompl",
            load_all=False,
        )
        .to_moveit_configs()
    )

    robot_description = ParameterValue(
        Command([
            "xacro",
            " ",
            PathJoinSubstitution([
                FindPackageShare("dsr_description2"),
                "xacro",
                LaunchConfiguration('model'),
            ]),
            ".urdf.xacro color:=",
            LaunchConfiguration('color'),
        ]),
        value_type=str,
    )

    return [
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            namespace=LaunchConfiguration('name'),
            output="screen",
            parameters=[
                moveit_config.to_dict(),
                {"robot_description": robot_description},
            ],
        )
    ]

def generate_launch_description():
    ARGUMENTS =[ 
        DeclareLaunchArgument('name',         default_value = 'dsr01',          description = 'NAME_SPACE'              ),
        DeclareLaunchArgument('host',         default_value = '127.0.0.1',      description = 'ROBOT_IP'                ),
        DeclareLaunchArgument('port',         default_value = '12345',          description = 'ROBOT_PORT'              ),
        DeclareLaunchArgument('mode',         default_value = 'virtual',        description = 'OPERATION MODE'          ),
        DeclareLaunchArgument('model',        default_value = 'm0609',          description = 'ROBOT_MODEL'             ),
        DeclareLaunchArgument('color',        default_value = 'white',          description = 'ROBOT_COLOR'             ),
        DeclareLaunchArgument('gui',          default_value = 'true',          description = 'Start RViz2'             ),
        DeclareLaunchArgument('gz',           default_value = 'true',           description = 'USE GAZEBO SIM'          ),
        DeclareLaunchArgument('x',            default_value = '-0.61',              description = 'Location x on Gazebo '   ),
        DeclareLaunchArgument('y',            default_value = '0.365',              description = 'Location y on Gazebo'    ),
        DeclareLaunchArgument('z',            default_value = '0.91',              description = 'Location z on Gazebo'    ),
        DeclareLaunchArgument('R',            default_value = '0',              description = 'Location Roll on Gazebo' ),
        DeclareLaunchArgument('P',            default_value = '0',              description = 'Location Pitch on Gazebo'),
        DeclareLaunchArgument('Y',            default_value = '3.141519',              description = 'Location Yaw on Gazebo'  ),
        DeclareLaunchArgument('rt_host',      default_value = '192.168.137.50', description = 'ROBOT_RT_IP'             ),
        DeclareLaunchArgument('gripper',      default_value = 'none',           description = 'GRIPPER'                 ),
        DeclareLaunchArgument('use_sim_time', default_value='false',            description='Use simulation time'       ),
        DeclareLaunchArgument('remap_tf',     default_value = 'false',          description = 'REMAP TF'                ),
        DeclareLaunchArgument(
            'aruco_markers',
            default_value='box_corners',
            description=(
                "Use 'box_corners' to place IDs 1-4 on the box corners, or provide "
                "semicolon-separated entries in 'id,x,y,z,roll,pitch,yaw' format. "
                "Use an empty value to disable them."
            ),
        ),
    ]
    
    set_use_sim_time = SetLaunchConfiguration(name='use_sim_time', value='false')
    xacro_path = os.path.join( get_package_share_directory('dsr_description2'), 'xacro')
    # Initialize Arguments
    gui = LaunchConfiguration("gui")
    mode = LaunchConfiguration("mode")
    update_rate = read_update_rate() # get update_rate from yaml
    show_git_info() # print git info
    # Get URDF via xacro
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [
                    FindPackageShare("dsr_description2"),
                    "xacro",
                    LaunchConfiguration('model'),
                ]
            ),
            ".urdf.xacro",
        ]
    )

    robot_description = {"robot_description": robot_description_content}

    robot_controllers = PathJoinSubstitution(
        [
            FindPackageShare("dsr_controller2"),
            "config",
            "dsr_controller2.yaml",
        ]
    )
    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare("dsr_description2"), "rviz", "default.rviz"]
    )
    
    set_config_node = Node(
        package="dsr_bringup2",
        executable="set_config",
        namespace=LaunchConfiguration('name'),
        parameters=[
            {"name":    LaunchConfiguration('name')  }, 
            {"rate":    100         },
            {"standby": 5000        },
            {"command": True        },
            {"host":    LaunchConfiguration('host')  },
            {"port":    LaunchConfiguration('port')  },
            {"mode":    LaunchConfiguration('mode')  },
            {"model":   LaunchConfiguration('model') },
            {"gripper": LaunchConfiguration('gripper')      },
            {"mobile":  "none"      },
            {"rt_host":  LaunchConfiguration('rt_host')      },
            {"update_rate": update_rate        },
            #parameters_file_path       # 파라미터 설정을 동일이름으로 launch 파일과 yaml 파일에서 할 경우 yaml 파일로 셋팅된다.    
        ],
        output="screen",
    )
    
    run_emulator_node = Node(
        package="dsr_bringup2",
        executable="run_emulator",
        namespace=LaunchConfiguration('name'),
        parameters=[
            {"name":    LaunchConfiguration('name')  }, 
            {"rate":    100         },
            {"standby": 5000        },
            {"command": True        },
            {"host":    LaunchConfiguration('host')  },
            {"port":    LaunchConfiguration('port')  },
            {"mode":    LaunchConfiguration('mode')  },
            {"model":   LaunchConfiguration('model') },
            {"gripper": LaunchConfiguration('gripper')      },
            {"mobile":  "none"      },
            {"rt_host":  LaunchConfiguration('rt_host')      },
            #parameters_file_path       # 파라미터 설정을 동일이름으로 launch 파일과 yaml 파일에서 할 경우 yaml 파일로 셋팅된다.    
        ],
        condition=IfCondition(PythonExpression(["'", mode, "' == 'virtual'"])),
        output="screen",
    )

    gazebo_connection_node = Node(
        package="dsr_bringup2",
        executable="gazebo_connection",
        namespace=LaunchConfiguration('name'),
        parameters=[
            {"model":   LaunchConfiguration('model') },
        ],
        output="log",
    )

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        namespace=LaunchConfiguration('name'),
        parameters=[robot_description, robot_controllers],
        # output="both",
    )
    
    robot_state_pub_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=LaunchConfiguration('name'),
        output='both',
        parameters=[{
            'robot_description': Command(['xacro', ' ', xacro_path, '/', LaunchConfiguration('model'), '.urdf.xacro color:=', LaunchConfiguration('color')])
        }],
    )
    
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        namespace=LaunchConfiguration('name'),
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        condition=IfCondition(gui),
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        namespace=LaunchConfiguration('name'),
        executable="spawner",
        arguments=["joint_state_broadcaster", "-c", "controller_manager"],
        parameters=[PathJoinSubstitution([FindPackageShare("dsr_controller2"), "config", "joint_state_broadcaster.yaml"])]
    )

    robot_controller_spawner = Node(
        package="controller_manager",
        namespace=LaunchConfiguration('name'),
        executable="spawner",
        arguments=["dsr_controller2", "-c", "controller_manager"],
    )

    dsr_moveit_controller_spawner = Node(
        package="controller_manager",
        namespace=LaunchConfiguration('name'),
        executable="spawner",
        arguments=["dsr_moveit_controller", "-c", "controller_manager"],
    )

    move_group_node = OpaqueFunction(function=move_group_fn)

    # Delay rviz start after `joint_state_broadcaster`
    delay_rviz_after_joint_state_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=robot_controller_spawner,
            on_exit=[rviz_node],
        )
    )

    original_tf_nodes = GroupAction(
        actions=[
            robot_state_pub_node,
            rviz_node
        ],
        condition=UnlessCondition(LaunchConfiguration('remap_tf'))
    )

    remapped_tf_nodes = GroupAction(
        actions=[
            SetRemap(src='/tf', dst='tf'),
            SetRemap(src='/tf_static', dst='tf_static'),
            robot_state_pub_node,
            rviz_node
        ],
        condition=IfCondition(LaunchConfiguration('remap_tf'))
    )


    #========= LAUNCH FILE THAT LOADS GAZEBO ELEMENTS ==========# 
    included_launch_file_path = os.path.join(
        get_package_share_directory('dsr_gazebo2'),
        'launch',
        'CNC_gazebo.launch.py'
    )
    
    included_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(included_launch_file_path),
        condition=IfCondition(LaunchConfiguration('gz')),
        launch_arguments={'use_gazebo': LaunchConfiguration('gz'), 
                          'name' : LaunchConfiguration('name'),
                          'model' : LaunchConfiguration('model'),
                          'color' : LaunchConfiguration('color'),
                          'gui' : 'false',
                          'x' :LaunchConfiguration('x'),
                          'y' :LaunchConfiguration('y'),
                          'z' :LaunchConfiguration('z'),
                          'R' :LaunchConfiguration('R'),
                          'P' :LaunchConfiguration('P'),
                          'Y' :LaunchConfiguration('Y'),
                          'use_sim_time' : LaunchConfiguration('use_sim_time'),
                          }.items(),
    )
    #========= LAUNCH FILE THAT LOADS GAZEBO ELEMENTS ==========# 

    delay_moveit_controller_after_robot_controller_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=robot_controller_spawner,
            on_exit=[
                LogInfo(msg=">> dsr_controller2 active. Starting dsr_moveit_controller..."),
                dsr_moveit_controller_spawner,
            ],
        )
    )

    # Delay start of robot_controller after `joint_state_broadcaster`
    delay_control_node_after_connection_node = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=set_config_node,
            on_exit=[control_node],
        )
    )

    nodes = [
        set_use_sim_time,
        set_config_node,
        run_emulator_node,
        gazebo_connection_node,
        LogInfo(msg=">> Starting MoveIt2 move_group for chip blowing..."),
        move_group_node,
        original_tf_nodes,
        remapped_tf_nodes,
        included_launch,
        OpaqueFunction(function=spawn_box_below_end_effector),
        OpaqueFunction(function=spawn_aruco_markers),
        robot_controller_spawner,
        joint_state_broadcaster_spawner,
        delay_moveit_controller_after_robot_controller_spawner,
        delay_control_node_after_connection_node,
    ]

    return LaunchDescription(ARGUMENTS + nodes)
