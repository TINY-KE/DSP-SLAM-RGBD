FIND_PACKAGE( PCL 1.10 REQUIRED )


add_library(Config SHARED
    Config.cpp
)

target_link_libraries(Config
        ${OpenCV_LIBS}
        )
