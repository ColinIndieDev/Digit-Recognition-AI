#pragma once

#include <functional>
#include "../CPL.h"
#include "Timer.h"

namespace CPL {
    struct Timer;

    class TimerManager {
    public:
        static void Update(float delta);
        static void AddTimer(float duration, bool loop, const std::function<void()>& cb);
        static void StopTimers();
        static void ClearTimers();
    private:
        static std::vector<Timer> timers;
    };
}