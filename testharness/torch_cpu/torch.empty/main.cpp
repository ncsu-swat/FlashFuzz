#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 2 bytes for basic parameters
        if (Size < 2) {
            return 0;
        }
        
        // Parse data type for the empty tensor
        uint8_t dtype_selector = Data[offset++];
        torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
        
        // Parse rank for the tensor shape
        uint8_t rank_byte = Data[offset++];
        uint8_t rank = fuzzer_utils::parseRank(rank_byte);
        
        // Parse shape dimensions
        std::vector<int64_t> shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
        
        // Create options with the selected data type
        auto options = torch::TensorOptions().dtype(dtype);
        
        // Create empty tensor with the specified shape and dtype
        torch::Tensor empty_tensor = torch::empty(shape, options);
        
        // Test some basic properties of the created tensor
        auto sizes = empty_tensor.sizes();
        auto strides = empty_tensor.strides();
        auto numel = empty_tensor.numel();
        auto tensor_dtype = empty_tensor.dtype();
        
        // Try to access some elements if the tensor is not empty
        if (numel > 0) {
            // For numeric types, we can try to access elements
            if (tensor_dtype == torch::kFloat || 
                tensor_dtype == torch::kDouble || 
                tensor_dtype == torch::kInt || 
                tensor_dtype == torch::kLong) {
                
                // Access first element (this might contain uninitialized data, which is expected)
                auto first_elem = empty_tensor.data_ptr();
                
                // Try to modify the tensor (should be allowed for empty)
                if (tensor_dtype == torch::kFloat) {
                    empty_tensor[0] = 1.0f;
                } else if (tensor_dtype == torch::kDouble) {
                    empty_tensor[0] = 1.0;
                } else if (tensor_dtype == torch::kInt || tensor_dtype == torch::kLong) {
                    empty_tensor[0] = 1;
                }
            }
        }
        
        // Test with different memory layouts
        if (Size > offset && numel > 0) {
            uint8_t layout_selector = Data[offset++];
            
            // Try different memory formats based on the selector
            if (layout_selector % 3 == 0) {
                // Test contiguous format
                empty_tensor = empty_tensor.contiguous();
            } else if (layout_selector % 3 == 1) {
                // Test transposed format (for 2D+ tensors)
                if (rank >= 2) {
                    empty_tensor = empty_tensor.transpose(0, 1);
                }
            } else {
                // Test strided format
                if (rank >= 1) {
                    empty_tensor = empty_tensor.as_strided(shape, strides);
                }
            }
        }
        
        // Test with pinned memory if available
        if (Size > offset) {
            uint8_t pin_selector = Data[offset++];
            if (pin_selector % 2 == 0) {
                try {
                    // Create empty tensor with pinned memory
                    auto pinned_options = options.pinned_memory(true);
                    torch::Tensor pinned_empty = torch::empty(shape, pinned_options);
                } catch (...) {
                    // Pinned memory might not be available, ignore errors
                }
            }
        }
        
        // Test with different device options if available
        if (Size > offset) {
            uint8_t device_selector = Data[offset++];
            if (device_selector % 2 == 0) {
                try {
                    // Try to create on CUDA device if available
                    auto cuda_options = options.device(torch::kCUDA);
                    torch::Tensor cuda_empty = torch::empty(shape, cuda_options);
                } catch (...) {
                    // CUDA might not be available, ignore errors
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}