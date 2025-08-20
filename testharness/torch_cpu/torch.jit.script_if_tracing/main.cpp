#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the fuzzer data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple function to be scripted
        auto fn = [](const torch::Tensor& x) -> torch::Tensor {
            return x.sin();
        };
        
        // Create a tracing state
        bool is_tracing = false;
        if (offset < Size) {
            is_tracing = Data[offset++] % 2 == 0;
        }
        
        // Create a function that will be used with script_if_tracing
        auto scripted_fn = [&](const torch::Tensor& x) -> torch::Tensor {
            if (torch::jit::tracer::isTracing()) {
                return x.sin();
            } else {
                return x.cos();
            }
        };
        
        // Apply the function directly
        torch::Tensor result;
        
        // Test the script_if_tracing functionality
        if (is_tracing) {
            // Simulate tracing environment
            try {
                auto scripted_module = torch::jit::script(fn);
                result = scripted_fn(input_tensor);
            } catch (...) {
                // Handle any exceptions from scripting
            }
        } else {
            // Regular execution (not tracing)
            result = scripted_fn(input_tensor);
        }
        
        // Try another variant with different tensor
        if (offset < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Create a more complex function for script_if_tracing
            auto complex_fn = [&](const torch::Tensor& x) -> torch::Tensor {
                if (torch::jit::tracer::isTracing()) {
                    return x.pow(2).log().abs();
                } else {
                    return x.exp().tanh();
                }
            };
            
            torch::Tensor complex_result = complex_fn(another_tensor);
        }
        
        // Test with edge case tensors
        if (offset + 1 < Size) {
            // Create a scalar tensor
            torch::Tensor scalar_tensor = torch::tensor(static_cast<float>(Data[offset]));
            
            auto scalar_fn = [&](const torch::Tensor& x) -> torch::Tensor {
                if (torch::jit::tracer::isTracing()) {
                    return x + 1;
                } else {
                    return x - 1;
                }
            };
            
            torch::Tensor scalar_result = scalar_fn(scalar_tensor);
            offset++;
        }
        
        // Test with empty tensor
        if (offset < Size) {
            std::vector<int64_t> empty_shape = {0};
            torch::Tensor empty_tensor = torch::empty(empty_shape);
            
            auto empty_fn = [&](const torch::Tensor& x) -> torch::Tensor {
                if (torch::jit::tracer::isTracing()) {
                    return torch::ones_like(x);
                } else {
                    return torch::zeros_like(x);
                }
            };
            
            torch::Tensor empty_result = empty_fn(empty_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}