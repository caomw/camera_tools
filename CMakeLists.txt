cmake_minimum_required(VERSION 2.8.3)
project(camera_tools)

add_definitions(-std=c++11 -Wall)

find_package(catkin REQUIRED COMPONENTS
    roscpp cv_bridge image_geometry image_transport message_filters sensor_msgs
    dynamic_reconfigure
    )

generate_dynamic_reconfigure_options(cfg/StereoCheckerDyn.cfg)

catkin_package(
    #  INCLUDE_DIRS include
    #  LIBRARIES camera_tool
    #  CATKIN_DEPENDS cv_bridge image_geometry message_filters sensor_msgs
    #  DEPENDS system_lib
    )

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    )

add_executable(stereo_checker
    src/stereo_checker.cpp
    )
target_link_libraries(stereo_checker
    ${catkin_LIBRARIES}
    )

add_dependencies(stereo_checker
    ${catkin_EXPORTED_TARGETS}
    ${PROJECT_NAME}_gencfg
    ${PROJECT_NAME}_gencpp
    ${${PROJECT_NAME}_EXPORTED_TARGETS}
    )
