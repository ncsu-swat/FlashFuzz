#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>
#include <torch/csrc/profiler/api.h>
#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/containers.h>
#include <vector>
#include <string>
#include <memory>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0; // Need minimum data for meaningful fuzzing
        }

        // Extract fuzzing parameters
        uint8_t profiler_mode = consume_uint8_t(Data, Size, offset);
        uint8_t record_shapes = consume_uint8_t(Data, Size, offset);
        uint8_t profile_memory = consume_uint8_t(Data, Size, offset);
        uint8_t with_stack = consume_uint8_t(Data, Size, offset);
        uint8_t with_flops = consume_uint8_t(Data, Size, offset);
        uint8_t with_modules = consume_uint8_t(Data, Size, offset);
        uint8_t experimental_config = consume_uint8_t(Data, Size, offset);
        uint8_t tensor_ops_count = consume_uint8_t(Data, Size, offset) % 10 + 1; // 1-10 operations
        
        // Create profiler configuration
        torch::profiler::ProfilerConfig config;
        
        // Set profiler state based on fuzzer input
        switch (profiler_mode % 4) {
            case 0:
                config.state = torch::profiler::ProfilerState::Disabled;
                break;
            case 1:
                config.state = torch::profiler::ProfilerState::CPU;
                break;
            case 2:
                config.state = torch::profiler::ProfilerState::CUDA;
                break;
            case 3:
                config.state = torch::profiler::ProfilerState::NVTX;
                break;
        }
        
        config.record_shapes = (record_shapes % 2) == 1;
        config.profile_memory = (profile_memory % 2) == 1;
        config.with_stack = (with_stack % 2) == 1;
        config.with_flops = (with_flops % 2) == 1;
        config.with_modules = (with_modules % 2) == 1;
        
        // Set experimental config flags
        if (experimental_config % 2 == 1) {
            config.experimental_config.verbose = true;
        }
        
        // Test profiler enable/disable cycle
        try {
            torch::profiler::impl::enableProfiler(config, {});
            
            // Perform some tensor operations while profiling
            for (uint8_t i = 0; i < tensor_ops_count && offset < Size; ++i) {
                uint8_t op_type = consume_uint8_t(Data, Size, offset);
                
                // Create test tensors with varying properties
                auto tensor_size = consume_uint8_t(Data, Size, offset) % 100 + 1;
                auto tensor1 = torch::randn({tensor_size, tensor_size});
                auto tensor2 = torch::randn({tensor_size, tensor_size});
                
                // Perform different operations based on fuzzer input
                switch (op_type % 8) {
                    case 0: {
                        auto result = tensor1 + tensor2;
                        break;
                    }
                    case 1: {
                        auto result = tensor1.matmul(tensor2);
                        break;
                    }
                    case 2: {
                        auto result = torch::relu(tensor1);
                        break;
                    }
                    case 3: {
                        auto result = torch::softmax(tensor1, 0);
                        break;
                    }
                    case 4: {
                        auto result = tensor1.sum();
                        break;
                    }
                    case 5: {
                        auto result = tensor1.transpose(0, 1);
                        break;
                    }
                    case 6: {
                        auto result = torch::conv2d(tensor1.unsqueeze(0).unsqueeze(0), 
                                                   tensor2.unsqueeze(0).unsqueeze(0));
                        break;
                    }
                    case 7: {
                        auto result = tensor1.view({-1});
                        break;
                    }
                }
            }
            
            // Test profiler result collection
            auto profiler_results = torch::profiler::impl::disableProfiler();
            
            if (profiler_results) {
                // Test various profiler result operations
                auto events = profiler_results->events();
                
                // Test event iteration and property access
                for (const auto& event : events) {
                    // Access event properties to test serialization/deserialization
                    auto name = event.name();
                    auto duration = event.duration_time_ns();
                    auto cpu_time = event.cpu_time_total();
                    
                    // Test shape information if available
                    if (config.record_shapes) {
                        auto shapes = event.shapes();
                    }
                    
                    // Test memory information if available
                    if (config.profile_memory) {
                        auto cpu_memory = event.cpu_memory_usage();
                        auto cuda_memory = event.cuda_memory_usage();
                    }
                    
                    // Test stack information if available
                    if (config.with_stack) {
                        auto stack = event.stack();
                    }
                    
                    // Test module information if available
                    if (config.with_modules) {
                        auto module = event.module_hierarchy();
                    }
                }
                
                // Test profiler table generation
                try {
                    auto table = profiler_results->table();
                } catch (...) {
                    // Table generation might fail in some configurations
                }
                
                // Test trace export functionality
                if (offset + 1 < Size) {
                    uint8_t export_format = consume_uint8_t(Data, Size, offset);
                    try {
                        if (export_format % 2 == 0) {
                            // Test Chrome trace export
                            auto trace = profiler_results->trace();
                        } else {
                            // Test other export formats if available
                            auto table_str = profiler_results->table();
                        }
                    } catch (...) {
                        // Export might fail in some configurations
                    }
                }
            }
            
        } catch (const std::exception& e) {
            // Profiler might not be available or fail to initialize
            // This is acceptable for fuzzing
        }
        
        // Test profiler utilities and edge cases
        if (offset < Size) {
            uint8_t utility_test = consume_uint8_t(Data, Size, offset);
            
            switch (utility_test % 4) {
                case 0: {
                    // Test profiler state queries
                    auto is_enabled = torch::profiler::impl::profilerEnabled();
                    break;
                }
                case 1: {
                    // Test record function scopes
                    torch::profiler::RecordProfile guard("test_scope");
                    auto dummy = torch::randn({10, 10});
                    break;
                }
                case 2: {
                    // Test nested profiler scopes
                    {
                        torch::profiler::RecordProfile outer_guard("outer_scope");
                        {
                            torch::profiler::RecordProfile inner_guard("inner_scope");
                            auto dummy = torch::ones({5, 5});
                        }
                    }
                    break;
                }
                case 3: {
                    // Test profiler with custom activities
                    torch::profiler::ProfilerConfig custom_config;
                    custom_config.state = torch::profiler::ProfilerState::CPU;
                    custom_config.record_shapes = true;
                    
                    try {
                        torch::profiler::impl::enableProfiler(custom_config, {});
                        auto tensor = torch::randn({20, 20});
                        auto result = tensor.sum();
                        torch::profiler::impl::disableProfiler();
                    } catch (...) {
                        // Custom config might fail
                    }
                    break;
                }
            }
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}