# set(CMAKE_BUILD_TYPE Debug)
FIND_PACKAGE( PCL 1.10 REQUIRED )

add_library(tum_rgbd SHARED
    io.cpp
)

target_link_libraries(tum_rgbd
        ${OpenCV_LIBS}
        utils
        )
