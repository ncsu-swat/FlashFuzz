#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr/cout
#include <cstdint>        // For uint64_t

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
        
        // Extract a dimension to use for the all operation if there's enough data
        int64_t dim = 0;
        bool keepdim = false;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // If tensor has dimensions, ensure dim is within valid range
            if (input_tensor.dim() > 0) {
                dim = dim % input_tensor.dim();
                if (dim < 0) {
                    dim += input_tensor.dim();
                }
            }
        }
        
        // Extract keepdim parameter if there's enough data
        if (offset < Size) {
            keepdim = static_cast<bool>(Data[offset] & 0x1);
            offset++;
        }
        
        // Test torch.all with different variants
        
        // Variant 1: all() - reduce over all dimensions
        torch::Tensor result1 = torch::all(input_tensor);
        (void)result1; // Suppress unused variable warning
        
        // Variant 2: all(dim, keepdim) - reduce over specific dimension
        if (input_tensor.dim() > 0) {
            try {
                torch::Tensor result2 = torch::all(input_tensor, dim, keepdim);
                (void)result2;
            } catch (const std::exception &) {
                // Shape/dimension errors are expected for some inputs
            }
        }
        
        // Variant 3: all(dim) - reduce over specific dimension without keepdim
        if (input_tensor.dim() > 0) {
            try {
                torch::Tensor result3 = torch::all(input_tensor, dim);
                (void)result3;
            } catch (const std::exception &) {
                // Shape/dimension errors are expected for some inputs
            }
        }
        
        // Variant 4: Test with boolean tensor explicitly
        try {
            // Convert to boolean tensor
            torch::Tensor bool_tensor = input_tensor.to(torch::kBool);
            torch::Tensor result4 = torch::all(bool_tensor);
            (void)result4;
            
            if (bool_tensor.dim() > 0) {
                torch::Tensor result5 = torch::all(bool_tensor, dim, keepdim);
                (void)result5;
            }
        } catch (const std::exception &) {
            // Conversion or dimension errors are expected for some inputs
        }
        
        // Variant 5: Test with different dtypes
        try {
            torch::Tensor int_tensor = input_tensor.to(torch::kInt);
            torch::Tensor result6 = torch::all(int_tensor);
            (void)result6;
        } catch (const std::exception &) {
            // Dtype conversion errors are expected for some inputs
        }
        
        // Variant 6: Test with contiguous/non-contiguous tensors
        if (input_tensor.dim() >= 2) {
            try {
                torch::Tensor transposed = input_tensor.transpose(0, 1);
                torch::Tensor result7 = torch::all(transposed);
                (void)result7;
            } catch (const std::exception &) {
                // Transposition errors are expected for some inputs
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}