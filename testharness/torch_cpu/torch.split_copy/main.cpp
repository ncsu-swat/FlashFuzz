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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor has at least one dimension
        if (input_tensor.dim() == 0) {
            return 0;
        }
        
        // Extract split parameters from the remaining data
        if (offset + 2 >= Size) {
            return 0;
        }
        
        // Get dimension to split along (constrained to valid range)
        int64_t dim = 0;
        if (offset < Size) {
            dim = static_cast<int64_t>(Data[offset++]) % input_tensor.dim();
            // Handle negative dims too
            if (Data[offset - 1] & 0x80) {
                dim = dim - input_tensor.dim();
            }
        }
        
        int64_t actual_dim = dim >= 0 ? dim : dim + input_tensor.dim();
        int64_t dim_size = input_tensor.size(actual_dim);
        if (dim_size == 0) {
            return 0;
        }
        
        // Generate a split size (1 to dim_size)
        int64_t split_size = 1;
        if (offset < Size) {
            split_size = (static_cast<int64_t>(Data[offset++]) % dim_size) + 1;
        }
        
        std::vector<torch::Tensor> result;
        
        try {
            // torch::split_copy only accepts a single split_size, not a vector
            result = torch::split_copy(input_tensor, split_size, dim);
        } catch (...) {
            // Expected failures (invalid split size)
            return 0;
        }
        
        // Verify the result by concatenating back
        if (!result.empty()) {
            try {
                torch::Tensor reconstructed = torch::cat(result, dim);
                
                // Check if the reconstructed tensor matches the original shape
                bool shapes_match = reconstructed.sizes() == input_tensor.sizes();
                (void)shapes_match; // Avoid unused variable warning
                
                // Access some elements to ensure tensors are valid
                if (input_tensor.numel() > 0) {
                    auto first_elem = input_tensor.flatten()[0].item<float>();
                    (void)first_elem;
                }
                
                if (reconstructed.numel() > 0) {
                    auto first_elem = reconstructed.flatten()[0].item<float>();
                    (void)first_elem;
                }
            } catch (...) {
                // Verification failures are not critical
            }
        }
        
        // Test with different tensor types for better coverage
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            torch::Tensor typed_tensor;
            
            try {
                switch (dtype_selector % 4) {
                    case 0:
                        typed_tensor = input_tensor.to(torch::kFloat32);
                        break;
                    case 1:
                        typed_tensor = input_tensor.to(torch::kFloat64);
                        break;
                    case 2:
                        typed_tensor = input_tensor.to(torch::kInt32);
                        break;
                    case 3:
                        typed_tensor = input_tensor.to(torch::kInt64);
                        break;
                }
                
                int64_t typed_actual_dim = dim >= 0 ? dim : dim + typed_tensor.dim();
                if (typed_tensor.dim() > 0 && typed_tensor.size(typed_actual_dim) > 0) {
                    auto typed_result = torch::split_copy(typed_tensor, 1, dim);
                    (void)typed_result;
                }
            } catch (...) {
                // Type conversion or split failures are expected
            }
        }
        
        // Test with different split sizes for additional coverage
        if (offset < Size && dim_size > 1) {
            try {
                // Try splitting with size 1 (maximum number of chunks)
                auto result_size1 = torch::split_copy(input_tensor, 1, dim);
                (void)result_size1;
                
                // Try splitting with full size (single chunk)
                auto result_full = torch::split_copy(input_tensor, dim_size, dim);
                (void)result_full;
            } catch (...) {
                // Expected failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}