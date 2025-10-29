#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#include "Audio.h"
#include "Logging.h"

namespace CPL {
    ma_engine AudioManager::engine;
    std::unique_ptr<ma_sound> AudioManager::music;
    std::vector<std::unique_ptr<ma_sound>> AudioManager::activeSounds;

    void AudioManager::Init() {
        if (ma_engine_init(nullptr, &engine) != MA_SUCCESS) {
            Logging::Log(2, "Failed to initialize audio engine!");
            exit(-1);
        }
    }

    Audio AudioManager::LoadAudio(const std::string& audioPath) {
        return {audioPath};
    }

    void AudioManager::Update() {
        std::erase_if(activeSounds, [](auto& s) {
            if (!ma_sound_is_playing(s.get())) {
                ma_sound_uninit(s.get());
                return true;
            }
            return false;
        });
    }

    void AudioManager::PlaySFX(const Audio& audio) {
        auto sound = std::make_unique<ma_sound>();
        if (ma_sound_init_from_file(&engine, audio.path.c_str(), MA_SOUND_FLAG_DECODE, nullptr, nullptr, sound.get()) != MA_SUCCESS) {
            Logging::Log(2, "Failed to initialize SFX!");
            return;
        }
        ma_sound_set_pitch(sound.get(), 1.0f);
        ma_sound_set_looping(sound.get(), MA_FALSE);
        ma_sound_start(sound.get());
        activeSounds.push_back(std::move(sound));
    }

    void AudioManager::PlaySFXPitch(const Audio& audio, const float pitch) {
        auto sound = std::make_unique<ma_sound>();
        if (ma_sound_init_from_file(&engine, audio.path.c_str(), MA_SOUND_FLAG_DECODE, nullptr, nullptr, sound.get()) != MA_SUCCESS) {
            Logging::Log(2, "Failed to initialize SFX!");
            return;
        }
        ma_sound_set_pitch(sound.get(), pitch);
        ma_sound_set_looping(sound.get(), MA_FALSE);
        ma_sound_start(sound.get());
        activeSounds.push_back(std::move(sound));
    }

    void AudioManager::PlayMusic(const Audio& audio) {
        if (music) {
            ma_sound_stop(music.get());
            ma_sound_uninit(music.get());
            music.reset();
        }

        music = std::make_unique<ma_sound>();
        if (ma_sound_init_from_file(&engine, audio.path.c_str(), MA_SOUND_FLAG_DECODE, nullptr, nullptr, music.get()) != MA_SUCCESS) {
            Logging::Log(2, "Failed to load music!");
            music.reset();
            return;
        }

        ma_sound_set_looping(music.get(), MA_TRUE);
        ma_sound_start(music.get());
    }

    void AudioManager::PauseMusic() {
        if (music) ma_sound_stop(music.get());
    }

    void AudioManager::ResumeMusic() {
        if (music) ma_sound_start(music.get());
    }

    void AudioManager::StopMusic() {
        if (music) {
            ma_sound_stop(music.get());
            ma_sound_seek_to_pcm_frame(music.get(), 0);
        }
    }

    void AudioManager::PlayMusicPitch(const Audio& audio, const float pitch) {
        const auto sound = new ma_sound;
        if (const ma_result result = ma_sound_init_from_file(&engine, audio.path.c_str(), MA_SOUND_FLAG_DECODE, nullptr, nullptr, sound);
            result != MA_SUCCESS) {
            Logging::Log(2, "Failed to initialize SFX!");
            return;
            }
        ma_sound_set_pitch(sound, pitch);
        ma_sound_set_looping(sound, MA_TRUE);
        ma_sound_start(sound);
    }

    void AudioManager::Close() {
        ma_engine_uninit(&engine);
    }
}
