cmake_minimum_required(VERSION 3.31)
project(AI)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# GLFW
include(FetchContent)
FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG master
)
FetchContent_MakeAvailable(glfw)

# GLM
FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG master
)
FetchContent_MakeAvailable(glm)

# Freetype
FetchContent_Declare(
        freetype
        GIT_REPOSITORY https://github.com/freetype/freetype.git
        GIT_TAG VER-2-14-1
)
FetchContent_MakeAvailable(freetype)

# STB
FetchContent_Declare(
        stb
        GIT_REPOSITORY https://github.com/nothings/stb.git
        GIT_TAG master
)
FetchContent_MakeAvailable(stb)

# Miniaudio
set(MINIAUDIO_PATH "${CMAKE_BINARY_DIR}/_deps/miniaudio/miniaudio.h")
file(DOWNLOAD
        https://raw.githubusercontent.com/mackron/miniaudio/master/miniaudio.h
        ${MINIAUDIO_PATH}
        SHOW_PROGRESS
)
add_library(miniaudio INTERFACE)
target_include_directories(miniaudio INTERFACE "${CMAKE_BINARY_DIR}/_deps/miniaudio")

# GLAD
add_subdirectory(external/glad)

add_executable(AI src/main.cpp
        src/NeuralNetwork.cpp
        src/NeuralNetwork.h
        src/MNISTloader.cpp
        src/MNISTloader.h
        src/TimerChrono.h

        CPLibrary/Shader.cpp
        CPLibrary/Shader.h
        CPLibrary/shapes2D/Triangle.cpp
        CPLibrary/shapes2D/Triangle.h
        CPLibrary/CPL.h
        CPLibrary/CPL.cpp
        CPLibrary/shapes2D/Rectangle.cpp
        CPLibrary/shapes2D/Rectangle.h
        CPLibrary/shapes2D/Circle.cpp
        CPLibrary/shapes2D/Circle.h
        CPLibrary/Colors.h
        CPLibrary/KeyInputs.h
        CPLibrary/Text.cpp
        CPLibrary/Text.h
        CPLibrary/shapes2D/Line.cpp
        CPLibrary/shapes2D/Line.h
        CPLibrary/shapes2D/Texture2D.cpp
        CPLibrary/shapes2D/Texture2D.h
        CPLibrary/timers/Timer.h
        CPLibrary/timers/TimerManager.cpp
        CPLibrary/timers/TimerManager.h
        CPLibrary/CPLibrary.h
        CPLibrary/Logging.h
        CPLibrary/Audio.cpp
        CPLibrary/Audio.h
        CPLibrary/Path.h
        src/CustomLoader.cpp
        src/CustomLoader.h
)

# Include directories
target_include_directories(AI PRIVATE ${stb_SOURCE_DIR})

# Link libraries
target_link_libraries(AI PRIVATE glfw glad freetype miniaudio glm)
