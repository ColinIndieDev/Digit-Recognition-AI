#pragma once
#include <filesystem>
#include <string>

class Path {
public:
    static std::string GetAssetPath(const std::string& relativePath) {
        // ----- IMPORTANT temporarily set manually path if release or not ----- //
        //return relativePath;
        return "../" + relativePath;
        // --------------------------------------------------------------------- //
    }
};