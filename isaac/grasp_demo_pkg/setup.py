from glob import glob
from setuptools import find_packages, setup

package_name = 'grasp_demo_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/weights', glob('weights/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Robotac Demo',
    maintainer_email='demo@example.com',
    description='Teaching demos for IsaacSim based YOLOE grasp and place modules.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'observation_pose_node = grasp_demo_pkg.observation_pose_node:main',
            'basic_arm_motion_node = grasp_demo_pkg.basic_arm_motion_node:main',
            'yoloe_detector_node = grasp_demo_pkg.yoloe_detector_node:main',
            'depth_pose_estimator_node = grasp_demo_pkg.depth_pose_estimator_node:main',
            'tf_transform_demo_node = grasp_demo_pkg.tf_transform_demo_node:main',
            'joint_state_filter_node = grasp_demo_pkg.joint_state_filter_node:main',
            'plan_to_pose_node = grasp_demo_pkg.plan_to_pose_node:main',
            'gripper_demo_node = grasp_demo_pkg.gripper_demo_node:main',
            'pick_place_state_machine = grasp_demo_pkg.pick_place_state_machine:main',
        ],
    },
)
