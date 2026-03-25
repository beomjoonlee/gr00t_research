from setuptools import find_packages, setup


package_name = "gr00t_piper_bridge"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/piper_bridge.launch.py"]),
    ],
    install_requires=["setuptools", "msgpack", "msgpack-numpy", "numpy"],
    zip_safe=True,
    maintainer="lbj",
    maintainer_email="lbj@example.com",
    description="ROS2 bridge node for GR00T and Piper integration.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "piper_bridge = gr00t_piper_bridge.piper_bridge_node:main",
        ],
    },
)
