#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <vector>         // For std::vector

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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Skip empty tensors
        if (input_tensor.numel() == 0) {
            return 0;
        }
        
        // Get a value to determine which variant to test
        uint8_t mode = 0;
        if (offset < Size) {
            mode = Data[offset++] % 3;
        }
        
        torch::Tensor result;
        
        if (mode == 0) {
            // Call ifftshift without dimension argument (shifts all dims)
            result = torch::fft::ifftshift(input_tensor);
        } else if (mode == 1) {
            // Call ifftshift with a single dimension
            if (offset < Size && input_tensor.dim() > 0) {
                int8_t dim_byte = static_cast<int8_t>(Data[offset++]);
                int64_t dim = dim_byte % input_tensor.dim();
                // Handle negative dimensions properly
                if (dim < 0) {
                    dim = dim % input_tensor.dim();
                }
                try {
                    result = torch::fft::ifftshift(input_tensor, dim);
                } catch (...) {
                    // Silently ignore invalid dim errors
                    return 0;
                }
            } else {
                result = torch::fft::ifftshift(input_tensor);
            }
        } else {
            // Call ifftshift with multiple dimensions
            if (offset < Size && input_tensor.dim() > 0) {
                uint8_t num_dims = (Data[offset++] % input_tensor.dim()) + 1;
                std::vector<int64_t> dims;
                for (uint8_t i = 0; i < num_dims && offset < Size; i++) {
                    int8_t dim_byte = static_cast<int8_t>(Data[offset++]);
                    int64_t dim = dim_byte % input_tensor.dim();
                    dims.push_back(dim);
                }
                if (!dims.empty()) {
                    try {
                        result = torch::fft::ifftshift(input_tensor, dims);
                    } catch (...) {
                        // Silently ignore invalid dim errors
                        return 0;
                    }
                } else {
                    result = torch::fft::ifftshift(input_tensor);
                }
            } else {
                result = torch::fft::ifftshift(input_tensor);
            }
        }
        
        // Force computation to ensure the operation is actually executed
        if (result.defined()) {
            volatile auto sum = result.sum().item<float>();
            (void)sum;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}