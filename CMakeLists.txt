cmake_minimum_required(VERSION 3.31)
project(AI)

set(CMAKE_CXX_STANDARD 20)

add_executable(AI main.cpp
        NeuralNetwork.cpp
        NeuralNetwork.h
        mnist_loader.cpp
        mnist_loader.h
        Timer.h)
