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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip empty tensors
        if (input_tensor.numel() == 0) {
            return 0;
        }
        
        // Extract dimension parameter for argmin if we have more data
        int64_t dim = 0;
        bool keepdim = false;
        uint8_t variant = 0;
        
        if (offset < Size) {
            variant = Data[offset] % 3;
            offset++;
        }
        
        if (offset < Size && input_tensor.dim() > 0) {
            // Extract dimension from the input data
            dim = static_cast<int64_t>(Data[offset] % input_tensor.dim());
            offset++;
        }
        
        // Extract keepdim parameter if we have more data
        if (offset < Size) {
            keepdim = static_cast<bool>(Data[offset] & 0x01);
            offset++;
        }
        
        // Apply argmin operation with different parameter combinations
        torch::Tensor result;
        
        // Test different variants of argmin
        if (variant == 0) {
            // Variant 1: argmin without parameters (reduces over all dimensions)
            result = torch::argmin(input_tensor);
        } 
        else if (variant == 1) {
            // Variant 2: argmin with dimension parameter
            try {
                result = torch::argmin(input_tensor, dim);
            } catch (const std::exception &) {
                // Shape mismatch or invalid dimension, try without dim
                result = torch::argmin(input_tensor);
            }
        }
        else {
            // Variant 3: argmin with dimension and keepdim parameters
            try {
                result = torch::argmin(input_tensor, dim, keepdim);
            } catch (const std::exception &) {
                // Shape mismatch or invalid dimension, try without dim
                result = torch::argmin(input_tensor);
            }
        }
        
        // Access result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            // Use data_ptr to access data without requiring single element
            auto *ptr = result.data_ptr<int64_t>();
            (void)ptr; // Prevent unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}