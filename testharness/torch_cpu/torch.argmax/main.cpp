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
        
        // Extract parameters for argmax from the remaining data
        int64_t dim = 0;
        bool keepdim = false;
        uint8_t variant = 0;
        
        // Parse variant selector
        if (offset < Size) {
            variant = Data[offset] % 3;
            offset++;
        }
        
        // Parse dimension parameter if we have more data
        if (offset < Size) {
            // Normalize dim to valid range for the tensor
            int64_t ndim = input_tensor.dim();
            if (ndim > 0) {
                dim = static_cast<int64_t>(Data[offset] % ndim);
                // Allow negative dimensions too
                if (offset + 1 < Size && (Data[offset + 1] & 0x01)) {
                    dim = -(ndim - dim);
                }
            }
            offset++;
        }
        
        // Parse keepdim parameter if we have more data
        if (offset < Size) {
            keepdim = static_cast<bool>(Data[offset] & 0x01);
            offset++;
        }
        
        torch::Tensor result;
        
        // Try different variants of argmax
        // Use inner try-catch for expected failures (don't log)
        if (variant == 0) {
            // argmax without dimension (finds max across all elements)
            try {
                result = torch::argmax(input_tensor);
            } catch (...) {
                return 0;
            }
        } else if (variant == 1) {
            // argmax with dimension
            try {
                result = torch::argmax(input_tensor, dim);
            } catch (...) {
                return 0;
            }
        } else {
            // argmax with dimension and keepdim
            try {
                result = torch::argmax(input_tensor, dim, keepdim);
            } catch (...) {
                return 0;
            }
        }
        
        // Perform some operations on the result to ensure it's used
        auto result_size = result.sizes();
        auto result_numel = result.numel();
        auto result_dtype = result.dtype();
        
        // Try to access elements if the result is a scalar
        if (result_numel == 1) {
            auto first_element = result.item<int64_t>();
            (void)first_element; // Suppress unused variable warning
        } else if (result_numel > 0) {
            // For non-scalar results, access via data_ptr
            auto data = result.data_ptr<int64_t>();
            (void)data;
        }
        
        // Additional coverage: test with different dtypes
        try {
            auto float_tensor = input_tensor.to(torch::kFloat32);
            auto float_result = torch::argmax(float_tensor);
            (void)float_result;
        } catch (...) {
            // Expected for some tensor types
        }
        
        // Test with contiguous tensor
        try {
            auto contig_tensor = input_tensor.contiguous();
            auto contig_result = torch::argmax(contig_tensor);
            (void)contig_result;
        } catch (...) {
            // Expected for some cases
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}