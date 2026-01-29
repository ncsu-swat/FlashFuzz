#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor - ensure it's a floating point type for ReLU6
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // ReLU6 requires floating point input
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create ReLU6 module with default options (inplace=false)
        torch::nn::ReLU6 relu6_module;
        
        // Apply ReLU6 operation
        torch::Tensor output = relu6_module->forward(input);
        
        // Also test the inplace variant
        if (Size > 10 && (Data[0] % 2 == 0)) {
            torch::nn::ReLU6 relu6_inplace(torch::nn::ReLU6Options().inplace(true));
            torch::Tensor input_copy = input.clone();
            relu6_inplace->forward(input_copy);
        }
        
        // Use the tensor in some way to prevent optimization
        if (output.defined() && output.numel() > 0) {
            auto sum = output.sum();
            // Access to ensure computation happens
            volatile float s = sum.item<float>();
            (void)s;
        }
        
        // Test functional form as well for better coverage
        torch::Tensor func_output = torch::nn::functional::relu6(input);
        
        if (func_output.defined() && func_output.numel() > 0) {
            volatile float s = func_output.sum().item<float>();
            (void)s;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}