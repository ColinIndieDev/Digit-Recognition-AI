#pragma once
#include <iostream>
#include <string>

namespace Logging {
    inline void Log(const int warnLevel, const std::string& text) {
        if (warnLevel == 0) std::cout << "[INFO]: " << text << std::endl;
        else if (warnLevel == 1) std::cout << "[WARNING]: " << text << std::endl;
        else std::cerr << "[ERROR]: " << text << std::endl;
    }
}