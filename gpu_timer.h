#pragma once
#ifndef OPINF_GPU_TIMER_H
#define OPINF_GPU_TIMER_H

#include <cuda_runtime_api.h>
#include <iostream>

template <bool isScopedTimer>
struct gpu_timer
{
    gpu_timer(std::string name) : event_name(name)
    {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
        if (isScopedTimer)
        {
            using std::cout;
            cout << name << " beginning." << std::endl;
            this->start();
        }
    }

    ~gpu_timer()
    {
        if (isScopedTimer)
        {
            this->stop();
            float elapsed_time = this->elapsed();
            std::cout << event_name << " ended after " << elapsed_time << std::endl;
        }
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start()
    {
        cudaEventRecord(start_event, 0);
    }

    void stop()
    {
        cudaEventRecord(stop_event, 0);
    }

    float elapsed()
    {
        float elapsed_dur;
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&elapsed_dur, start_event, stop_event);
        if (!isScopedTimer)
        {
            std::cout << "Elapsed time " << elapsed_dur << "ms" << std::endl;
        }

        return elapsed_dur;
    }
private:
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    std::string event_name;
};

#endif OPINF_GPU_TIMER_H