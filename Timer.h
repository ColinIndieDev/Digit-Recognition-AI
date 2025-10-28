#pragma once
#include <chrono>
#include <iostream>

struct Timer {
    std::chrono::time_point<std::chrono::steady_clock> start, end;
    std::chrono::duration<float> duration{};
    std::string text;

    explicit Timer(std::string text) : text(std::move(text)) {
        start = std::chrono::steady_clock::now();
    }

    ~Timer() {
        end = std::chrono::steady_clock::now();
        duration = end - start;
        const float ms = duration.count() * 1000.0f;
        std::cout << "------------------------------------------------------\n";
        std::cout << text.c_str() << " " << ms << " ms!\n";
        std::cout << "------------------------------------------------------\n";
    }
};