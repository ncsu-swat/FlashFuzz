#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }

        uint8_t profiler_config = Data[offset++];
        uint8_t activity_config = Data[offset++];
        uint8_t schedule_config = Data[offset++];
        uint8_t record_shapes = Data[offset++];
        uint8_t profile_memory = Data[offset++];
        uint8_t with_stack = Data[offset++];
        uint8_t with_flops = Data[offset++];
        uint8_t with_modules = Data[offset++];
        
        torch::profiler::ProfilerConfig config;
        
        std::set<torch::profiler::ProfilerActivity> activities;
        if (activity_config & 0x01) {
            activities.insert(torch::profiler::ProfilerActivity::CPU);
        }
        if (activity_config & 0x02) {
            activities.insert(torch::profiler::ProfilerActivity::CUDA);
        }
        
        config = torch::profiler::ProfilerConfig(
            activities,
            record_shapes % 2 == 1,
            profile_memory % 2 == 1,
            with_stack % 2 == 1,
            with_flops % 2 == 1,
            with_modules % 2 == 1
        );
        
        torch::profiler::ProfilerState state = static_cast<torch::profiler::ProfilerState>(profiler_config % 4);
        
        auto schedule_fn = [schedule_config](int step) -> torch::profiler::ProfilerAction {
            switch (schedule_config % 4) {
                case 0: return torch::profiler::ProfilerAction::NONE;
                case 1: return torch::profiler::ProfilerAction::WARMUP;
                case 2: return torch::profiler::ProfilerAction::RECORD;
                case 3: return torch::profiler::ProfilerAction::RECORD_AND_SAVE;
                default: return torch::profiler::ProfilerAction::NONE;
            }
        };
        
        torch::profiler::schedule schedule(schedule_fn);
        
        auto profiler = torch::profiler::profile(config);
        
        if (offset < Size) {
            auto tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
            
            profiler->start();
            
            auto result1 = tensor1 + 1.0f;
            auto result2 = torch::relu(result1);
            auto result3 = torch::sigmoid(result2);
            
            if (offset < Size) {
                auto tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                auto matmul_result = torch::matmul(tensor1.flatten(), tensor2.flatten().t());
                auto sum_result = torch::sum(matmul_result);
            }
            
            profiler->stop();
        }
        
        auto kineto_events = profiler->kineto_events();
        
        auto profiler_with_schedule = torch::profiler::profile(
            torch::profiler::ProfilerConfig(activities),
            schedule
        );
        
        for (int step = 0; step < 5; ++step) {
            profiler_with_schedule->step();
            
            if (offset < Size) {
                auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
                auto conv_result = torch::conv1d(tensor.unsqueeze(0), torch::ones({1, 1, 3}));
                auto pool_result = torch::avg_pool1d(conv_result, 2);
            }
        }
        
        auto events = profiler_with_schedule->events();
        
        torch::profiler::ProfilerGuard guard(config);
        
        if (offset < Size) {
            auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
            auto norm_result = torch::norm(tensor);
            auto exp_result = torch::exp(tensor);
            auto log_result = torch::log(torch::abs(tensor) + 1e-8);
        }
        
        auto memory_profiler = torch::profiler::profile(
            torch::profiler::ProfilerConfig(
                {torch::profiler::ProfilerActivity::CPU},
                false,
                true,
                false,
                false,
                false
            )
        );
        
        memory_profiler->start();
        
        if (offset < Size) {
            auto large_tensor = torch::randn({100, 100});
            auto allocated_tensor = torch::zeros({50, 50});
            auto copied_tensor = large_tensor.clone();
        }
        
        memory_profiler->stop();
        
        auto memory_events = memory_profiler->kineto_events();
        
        if (Size > 20) {
            torch::profiler::RecordFunction record("custom_function");
            
            if (offset < Size) {
                auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
                auto custom_op = tensor * 2.0f + torch::randn_like(tensor);
            }
        }
        
        auto nested_profiler = torch::profiler::profile(
            torch::profiler::ProfilerConfig({torch::profiler::ProfilerActivity::CPU})
        );
        
        nested_profiler->start();
        
        {
            torch::profiler::RecordFunction outer_record("outer_function");
            
            if (offset < Size) {
                auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                {
                    torch::profiler::RecordFunction inner_record("inner_function");
                    auto inner_result = torch::pow(tensor, 2.0f);
                    auto sqrt_result = torch::sqrt(torch::abs(inner_result) + 1e-8);
                }
                
                auto outer_result = torch::mean(tensor);
            }
        }
        
        nested_profiler->stop();
        
        if (profiler_config % 8 == 0) {
            auto export_profiler = torch::profiler::profile(
                torch::profiler::ProfilerConfig(
                    {torch::profiler::ProfilerActivity::CPU},
                    true,
                    true,
                    true,
                    true,
                    true
                )
            );
            
            export_profiler->start();
            
            if (offset < Size) {
                auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
                auto complex_op = torch::fft_fft(tensor.to(torch::kComplexFloat));
                auto real_part = torch::real(complex_op);
            }
            
            export_profiler->stop();
            
            try {
                export_profiler->export_chrome_trace("/tmp/trace.json");
            } catch (...) {
            }
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}