#pragma once

#include "miniaudio.h"
#include <string>
#include "CPL.h"
#include <memory>

namespace CPL {
    struct Audio {
        std::string path;
    };

    class AudioManager {
    public:
        static void Init();
        static void Update();
        static Audio LoadAudio(const std::string& audioPath);

        static void PlaySFX(const Audio& audio);
        static void PlaySFXPitch(const Audio& audio, float pitch);
        static void PlayMusic(const Audio& audio);
        static void PlayMusicPitch(const Audio& audio, float pitch);
        static void Close();
        static void PauseMusic();
        static void ResumeMusic();
        static void StopMusic();
    private:
        static ma_engine engine;
        static std::unique_ptr<ma_sound> music;
        static std::vector<std::unique_ptr<ma_sound>> activeSounds;
    };
}