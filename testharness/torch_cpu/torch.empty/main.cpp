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
                
                // Access data pointer (this might contain uninitialized data, which is expected)
                auto first_elem = empty_tensor.data_ptr();
                (void)first_elem; // Suppress unused variable warning
                
                // Flatten the tensor to safely access first element
                auto flat_tensor = empty_tensor.flatten();
                
                // Try to modify the tensor (should be allowed for empty)
                if (tensor_dtype == torch::kFloat) {
                    flat_tensor[0] = 1.0f;
                } else if (tensor_dtype == torch::kDouble) {
                    flat_tensor[0] = 1.0;
                } else if (tensor_dtype == torch::kInt || tensor_dtype == torch::kLong) {
                    flat_tensor[0] = 1;
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
                    try {
                        empty_tensor = empty_tensor.transpose(0, 1);
                    } catch (...) {
                        // Shape might not allow transpose, silently ignore
                    }
                }
            } else {
                // Test strided format
                if (rank >= 1) {
                    try {
                        empty_tensor = empty_tensor.as_strided(shape, strides);
                    } catch (...) {
                        // Strided access might fail, silently ignore
                    }
                }
            }
        }
        
        // Test with requires_grad option
        if (Size > offset) {
            uint8_t grad_selector = Data[offset++];
            if (grad_selector % 2 == 0 && 
                (dtype == torch::kFloat || dtype == torch::kDouble)) {
                auto grad_options = options.requires_grad(true);
                torch::Tensor grad_tensor = torch::empty(shape, grad_options);
                (void)grad_tensor.requires_grad(); // Access property
            }
        }
        
        // Test empty_like
        if (Size > offset) {
            uint8_t like_selector = Data[offset++];
            if (like_selector % 2 == 0) {
                torch::Tensor like_tensor = torch::empty_like(empty_tensor);
                (void)like_tensor.numel();
            }
        }
        
        // Test empty with specific memory format for 4D tensors
        if (rank == 4 && numel > 0) {
            try {
                torch::Tensor channels_last = torch::empty(
                    shape, 
                    options.memory_format(torch::MemoryFormat::ChannelsLast)
                );
                (void)channels_last.is_contiguous(torch::MemoryFormat::ChannelsLast);
            } catch (...) {
                // Memory format might not be supported for this shape
            }
        }
        
        // Test empty_strided
        if (Size > offset && rank > 0) {
            // Create valid strides for the shape
            std::vector<int64_t> custom_strides(rank);
            int64_t stride = 1;
            for (int i = rank - 1; i >= 0; --i) {
                custom_strides[i] = stride;
                stride *= (shape[i] > 0 ? shape[i] : 1);
            }
            try {
                torch::Tensor strided_tensor = torch::empty_strided(shape, custom_strides, options);
                (void)strided_tensor.strides();
            } catch (...) {
                // Strided creation might fail for some configurations
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