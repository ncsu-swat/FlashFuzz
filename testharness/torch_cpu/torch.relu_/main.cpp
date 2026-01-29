#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply relu_ in-place operation
        input_tensor.relu_();
        
        // Try with different tensor types if we have more data
        if (offset < Size) {
            size_t offset2 = 0;
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data + offset, Size - offset, offset2);
            another_tensor.relu_();
        }
        
        // Try with empty tensor
        try {
            torch::Tensor empty_tensor = torch::empty({0});
            empty_tensor.relu_();
        } catch (...) {
            // Expected to potentially fail, ignore
        }
        
        // Try with tensor containing extreme values
        std::vector<float> extreme_values = {
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::lowest(),
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN(),
            0.0f,
            -0.0f,
            1.0f,
            -1.0f
        };
        
        torch::Tensor extreme_tensor = torch::tensor(extreme_values);
        extreme_tensor.relu_();
        
        // Try with different dimensions
        if (Size >= 4) {
            uint8_t dim_byte = Data[0] % 4;
            std::vector<int64_t> shape;
            switch (dim_byte) {
                case 0: shape = {4}; break;
                case 1: shape = {2, 2}; break;
                case 2: shape = {2, 2, 1}; break;
                case 3: shape = {1, 2, 2, 1}; break;
            }
            torch::Tensor shaped_tensor = torch::randn(shape);
            shaped_tensor.relu_();
        }
        
        // Test with requires_grad tensor (in-place on leaf requires clone)
        try {
            torch::Tensor grad_tensor = torch::randn({3, 3}).clone().detach();
            grad_tensor.relu_();
        } catch (...) {
            // May fail due to autograd restrictions, ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}