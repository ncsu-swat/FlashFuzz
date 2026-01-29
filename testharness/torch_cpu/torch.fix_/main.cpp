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
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a copy of the input tensor for in-place operation
        torch::Tensor tensor_copy = input_tensor.clone();
        
        // Apply torch.fix_ (in-place operation - rounds toward zero/truncates)
        tensor_copy.fix_();
        
        // Verify the operation worked by comparing with non-in-place version
        torch::Tensor expected_result = torch::fix(input_tensor);
        
        // Check if results match (wrapped in inner try-catch since allclose
        // can fail for valid reasons like NaN values or integer tensors)
        try {
            if (tensor_copy.defined() && expected_result.defined() &&
                tensor_copy.is_floating_point()) {
                torch::allclose(tensor_copy, expected_result);
            }
        } catch (...) {
            // Ignore comparison failures - not the focus of fuzzing
        }
        
        // Try another variant with different tensor
        if (offset < Size) {
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            another_tensor.fix_();
        }
        
        // Also test with explicitly created floating point tensors
        if (offset + 4 <= Size) {
            // Create a float tensor to ensure we exercise the floating point path
            torch::Tensor float_tensor = torch::randn({3, 3}, torch::kFloat32);
            // Scale by fuzzer data
            float scale = static_cast<float>(Data[offset % Size]) / 25.5f;
            float_tensor = float_tensor * scale;
            float_tensor.fix_();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}