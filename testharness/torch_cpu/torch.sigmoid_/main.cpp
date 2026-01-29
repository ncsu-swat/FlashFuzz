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
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // sigmoid_ requires floating point tensor
        // Convert to float if not already a floating type
        if (!tensor.is_floating_point()) {
            tensor = tensor.to(torch::kFloat32);
        }
        
        // Apply the sigmoid_ operation in-place
        // sigmoid(x) = 1 / (1 + exp(-x))
        tensor.sigmoid_();
        
        // Access the result to ensure computation is performed
        if (tensor.defined() && tensor.numel() > 0) {
            volatile float first = tensor.flatten()[0].item<float>();
            (void)first;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}