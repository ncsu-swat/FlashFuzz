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
        uint8_t activities_mask = Data[offset++];
        uint8_t record_shapes = Data[offset++] % 2;
        uint8_t profile_memory = Data[offset++] % 2;
        uint8_t with_stack = Data[offset++] % 2;
        uint8_t with_flops = Data[offset++] % 2;
        uint8_t with_modules = Data[offset++] % 2;
        uint8_t use_cuda = Data[offset++] % 2;
        
        std::set<torch::profiler::ActivityType> activities;
        if (activities_mask & 0x01) {
            activities.insert(torch::profiler::ActivityType::CPU);
        }
        if (activities_mask & 0x02) {
            activities.insert(torch::profiler::ActivityType::CUDA);
        }
        
        if (activities.empty()) {
            activities.insert(torch::profiler::ActivityType::CPU);
        }

        auto config = torch::profiler::ProfilerConfig()
            .activities(activities)
            .record_shapes(record_shapes != 0)
            .profile_memory(profile_memory != 0)
            .with_stack(with_stack != 0)
            .with_flops(with_flops != 0)
            .with_modules(with_modules != 0);

        torch::profiler::enableProfiler(config);

        try {
            auto tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
            
            if (offset < Size) {
                auto tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                
                auto result1 = tensor1 + tensor2;
                auto result2 = torch::matmul(tensor1.flatten(), tensor2.flatten().t());
                auto result3 = torch::relu(tensor1);
                auto result4 = torch::sigmoid(tensor2);
                
                if (tensor1.numel() > 0 && tensor2.numel() > 0) {
                    auto result5 = torch::conv1d(tensor1.unsqueeze(0).unsqueeze(0), 
                                                tensor2.unsqueeze(0).unsqueeze(0));
                }
                
                auto result6 = torch::sum(tensor1);
                auto result7 = torch::mean(tensor2);
                
                if (tensor1.dim() > 0) {
                    auto result8 = torch::softmax(tensor1, 0);
                }
                
                auto result9 = torch::abs(tensor1);
                auto result10 = torch::exp(tensor2);
            } else {
                auto result1 = torch::sin(tensor1);
                auto result2 = torch::cos(tensor1);
                auto result3 = torch::tanh(tensor1);
                auto result4 = torch::sqrt(torch::abs(tensor1));
                auto result5 = torch::log(torch::abs(tensor1) + 1e-8);
                
                if (tensor1.numel() > 1) {
                    auto result6 = torch::norm(tensor1);
                    auto result7 = torch::var(tensor1);
                    auto result8 = torch::std(tensor1);
                }
            }
            
        } catch (const std::exception& e) {
            torch::profiler::disableProfiler();
            throw;
        }

        auto profiler_results = torch::profiler::disableProfiler();
        
        if (profiler_results) {
            auto events = profiler_results->events();
            
            for (const auto& event : events) {
                auto name = event.name();
                auto duration = event.cpu_elapsed_us();
                auto memory_usage = event.cpu_memory_usage();
                auto cuda_duration = event.cuda_elapsed_us();
                auto shapes = event.shapes();
                auto stack = event.stack();
                auto flops = event.flops();
            }
            
            auto table = profiler_results->table();
            auto trace = profiler_results->trace();
        }

        torch::profiler::ProfilerConfig nested_config;
        nested_config.activities({torch::profiler::ActivityType::CPU});
        
        torch::profiler::enableProfiler(nested_config);
        
        try {
            if (offset < Size) {
                auto tensor3 = fuzzer_utils::createTensor(Data, Size, offset);
                auto nested_result = torch::pow(tensor3, 2);
            }
        } catch (...) {
        }
        
        torch::profiler::disableProfiler();

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}