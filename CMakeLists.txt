cmake_policy(SET CMP0048 NEW)
project("Active-DSP-SLAM")
cmake_minimum_required(VERSION 3.2)

# c++ version
set(CMAKE_CXX_STANDARD 17)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()
SET(CMAKE_BUILD_TYPE Debug)

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

set(OpenCV_DIR "/home/zhjd/work/DSP-SLAM//Thirdparty/opencv/build")
set(Eigen3_DIR "/home/zhjd/work/DSP-SLAM//Thirdparty/eigen/install/share/eigen3/cmake")
set(Pangolin_DIR "/home/zhjd/work/DSP-SLAM//Thirdparty/Pangolin/build")
set(PYTHON_LIBRARIES "/home/zhjd/anaconda3/envs/dsp-slam/lib/libpython3.7m.so")
set(PYTHON_INCLUDE_DIRS "/home/zhjd/anaconda3/envs/dsp-slam/include/python3.7m")
set(PYTHON_EXECUTABLE "/home/zhjd/anaconda3/envs/dsp-slam/bin/python3.7")

#for ros
find_package(catkin REQUIRED COMPONENTS
        roscpp
        std_msgs
        visualization_msgs
        geometry_msgs
        sensor_msgs
        cv_bridge
        tf
)
find_package(OpenCV 3.2 REQUIRED)
find_package(Eigen3 3.4.0 REQUIRED)
find_package(Pangolin REQUIRED)
FIND_PACKAGE( PCL REQUIRED )
# 输出PCL的地址
message("PCL_INCLUDE_DIRS: ${PCL_INCLUDE_DIRS}")
message("PCL_LIBRARY_DIRS: ${PCL_LIBRARY_DIRS}")
message("PCL_LIBRARIES: ${PCL_LIBRARIES}")

#for ros
catkin_package(
#        INCLUDE_DIRS include
#        LIBRARIES path_planning
#        CATKIN_DEPENDS roscpp std_msgs visualization_msgs
#        DEPENDS system_lib
)

include_directories(
        ${catkin_INCLUDE_DIRS}  #for ros
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        /home/zhjd/work/DSP-SLAM/Thirdparty/g2o
        /home/zhjd/work/DSP-SLAM/Thirdparty/DBoW2
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
)


# You need to set to your own Python paths
add_subdirectory(pybind11)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
add_library(${PROJECT_NAME} SHARED
        src/System.cc
        src/System_util.cc
        src/Tracking.cc
        src/Tracking_util.cc
        src/LocalMapping.cc
        src/LocalMapping_util.cc
        src/LoopClosing.cc
        src/LoopClosing_util.cc
        src/ORBextractor.cc
        src/ORBmatcher.cc
        src/FrameDrawer.cc
        src/Converter.cc
        src/MapPoint.cc
        src/MapPoint_util.cc
        src/KeyFrame.cc
        src/KeyFrame_util.cc
        src/Map.cc
        src/Map_util.cc
        src/MapDrawer.cc
        src/Optimizer.cc
        src/Optimizer_util.cc
        src/PnPsolver.cc
        src/Frame.cc
        src/KeyFrameDatabase.cc
        src/Sim3Solver.cc
        src/Initializer.cc
        src/Viewer.cc
        src/ObjectDetection.cc
        src/MapObject.cc
        src/ObjectDrawer.cc
        src/ObjectRenderer.cc
        src/utils.cpp
        src/MapObject_util.cc
        src/Candidate.cpp

        # NBV
        #rrt的cpp文件x
        src/rrt.cpp
        src/obstacles.cpp
        src/environment.cpp
        src/NbvGenerator.cpp

        src/MapPublisher.cc

)
target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}  #for ros
        pybind11::embed
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
        /home/zhjd/work/DSP-SLAM/Thirdparty/DBoW2/lib/libDBoW2.so
        /home/zhjd/work/DSP-SLAM/Thirdparty/g2o/lib/libg2o.so
        -ltiff
        )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
add_executable(dsp_slam dsp_slam.cc)
target_link_libraries(dsp_slam ${PROJECT_NAME} -ltiff
        ${catkin_LIBRARIES}  #for ros
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
add_executable(dsp_slam_mono dsp_slam_mono.cc)
target_link_libraries(dsp_slam_mono ${PROJECT_NAME} -ltiff
        ${catkin_LIBRARIES}  #for ros
)

