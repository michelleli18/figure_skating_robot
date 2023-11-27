from setuptools import find_packages, setup
from glob import glob

package_name = 'figure_skating_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/urdf', glob('urdf/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'PreliminaryTesting = figure_skating_robot.PreliminaryTesting:main',
        	'GeneratorNode = figure_skating_robot.GeneratorNode:main',
        	'KinematicChain = figure_skating_robot.KinematicChain:main',
        	'TrajectoryUtils = figure_skating_robot.TrajectoryUtils:main',
        	'TransformHelpers = figure_skating_robot.TransformHelpers:main',
        ],
    },
)
