#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>
#include <torch/profiler.h>
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
        uint8_t profiler_config = consume_uint8_t(Data, Size, offset);
        uint8_t activity_config = consume_uint8_t(Data, Size, offset);
        uint8_t schedule_config = consume_uint8_t(Data, Size, offset);
        uint8_t tensor_ops = consume_uint8_t(Data, Size, offset);
        uint16_t warmup_steps = consume_uint16_t(Data, Size, offset);
        uint16_t active_steps = consume_uint16_t(Data, Size, offset);
        
        // Limit steps to reasonable ranges to avoid excessive execution time
        warmup_steps = warmup_steps % 5 + 1;
        active_steps = active_steps % 10 + 1;
        
        // Configure profiler activities
        std::set<torch::profiler::ActivityType> activities;
        if (activity_config & 0x01) {
            activities.insert(torch::profiler::ActivityType::CPU);
        }
        if (activity_config & 0x02) {
            activities.insert(torch::profiler::ActivityType::CUDA);
        }
        
        // Create profiler schedule
        auto schedule = torch::profiler::ProfilerSchedule::make_schedule(
            warmup_steps,
            active_steps,
            1, // repeat
            0  // skip_first
        );
        
        // Configure profiler config
        torch::profiler::ProfilerConfig config(
            activities,
            (profiler_config & 0x01) != 0, // record_shapes
            (profiler_config & 0x02) != 0, // profile_memory
            (profiler_config & 0x04) != 0, // with_stack
            (profiler_config & 0x08) != 0, // with_flops
            (profiler_config & 0x10) != 0, // with_modules
            schedule
        );
        
        // Create profiler
        auto profiler = std::make_unique<torch::profiler::ProfilerSession>(config);
        
        // Start profiling
        profiler->start();
        
        // Perform some tensor operations based on fuzzer input
        for (int step = 0; step < warmup_steps + active_steps; ++step) {
            profiler->step();
            
            // Create tensors with varying sizes based on fuzzer input
            int64_t tensor_size = (tensor_ops % 100) + 1;
            auto tensor1 = torch::randn({tensor_size, tensor_size});
            auto tensor2 = torch::randn({tensor_size, tensor_size});
            
            // Perform different operations based on fuzzer input
            switch (tensor_ops % 8) {
                case 0: {
                    auto result = tensor1 + tensor2;
                    break;
                }
                case 1: {
                    auto result = torch::mm(tensor1, tensor2);
                    break;
                }
                case 2: {
                    auto result = torch::relu(tensor1);
                    break;
                }
                case 3: {
                    auto result = torch::softmax(tensor1, 1);
                    break;
                }
                case 4: {
                    auto result = tensor1 * tensor2;
                    break;
                }
                case 5: {
                    auto result = torch::sum(tensor1);
                    break;
                }
                case 6: {
                    auto result = torch::transpose(tensor1, 0, 1);
                    break;
                }
                case 7: {
                    auto result = torch::sigmoid(tensor1);
                    break;
                }
            }
            
            // Add some nested profiling scopes
            {
                torch::profiler::RecordScope record_scope("custom_scope_1");
                auto temp = torch::ones({10, 10});
                temp = temp * 2.0;
            }
            
            if (step % 2 == 0) {
                torch::profiler::RecordScope record_scope("custom_scope_2");
                auto temp = torch::zeros({5, 5});
                temp = torch::sin(temp);
            }
            
            tensor_ops = (tensor_ops + 1) % 256; // Vary operations
        }
        
        // Stop profiling and get trace
        profiler->stop();
        
        // Try to export trace in different formats based on fuzzer input
        if (schedule_config & 0x01) {
            try {
                auto trace = profiler->trace();
                if (trace) {
                    // Test trace methods
                    trace->save("/tmp/test_trace.json");
                }
            } catch (...) {
                // Ignore export errors as they might be environment-dependent
            }
        }
        
        // Test profiler state transitions
        if (schedule_config & 0x02) {
            // Create another profiler session
            auto profiler2 = std::make_unique<torch::profiler::ProfilerSession>(config);
            profiler2->start();
            
            // Quick operation
            auto quick_tensor = torch::randn({2, 2});
            quick_tensor = quick_tensor + 1.0;
            
            profiler2->step();
            profiler2->stop();
        }
        
        // Test record function API
        if (schedule_config & 0x04) {
            torch::profiler::RecordScope record("fuzz_test_function");
            auto test_tensor = torch::randn({3, 3});
            test_tensor = torch::tanh(test_tensor);
        }
        
        // Test with different tensor types and devices
        if (schedule_config & 0x08) {
            auto int_tensor = torch::randint(0, 100, {5, 5}, torch::kInt32);
            auto float_tensor = int_tensor.to(torch::kFloat32);
            auto result = torch::sum(float_tensor);
        }
        
        // Test memory profiling scenarios
        if (profiler_config & 0x02) {
            std::vector<torch::Tensor> tensors;
            for (int i = 0; i < 5; ++i) {
                tensors.push_back(torch::randn({10, 10}));
            }
            // Let tensors go out of scope to trigger memory events
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}