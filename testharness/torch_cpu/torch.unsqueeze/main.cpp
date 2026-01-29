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
        
        // Need at least 1 byte for the dimension parameter
        if (Size < 1) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get dimension parameter for unsqueeze
        int64_t dim = 0;
        if (offset < Size) {
            // Extract a byte for the dimension
            uint8_t dim_byte = Data[offset++];
            
            // Convert to a dimension value that could be positive or negative
            int64_t rank = input_tensor.dim();
            
            // Valid dim range for unsqueeze is [-rank-1, rank]
            // Map dim_byte to this range
            int64_t range_size = 2 * rank + 2;
            if (range_size > 0) {
                dim = static_cast<int64_t>(dim_byte % range_size) - (rank + 1);
            }
        }
        
        // Apply unsqueeze operation
        torch::Tensor result = torch::unsqueeze(input_tensor, dim);
        
        // Verify the result has one more dimension than the input
        if (result.dim() != input_tensor.dim() + 1) {
            throw std::runtime_error("Unexpected result dimension");
        }
        
        // Access the data to ensure tensor is valid (sum works for any tensor)
        if (result.numel() > 0) {
            auto sum_val = result.sum();
            (void)sum_val;
        }
        
        // Try alternative API (method form)
        try {
            torch::Tensor result2 = input_tensor.unsqueeze(dim);
            
            // Verify both APIs produce same shape
            if (!result.sizes().equals(result2.sizes())) {
                throw std::runtime_error("API mismatch in result shapes");
            }
        } catch (const c10::Error &) {
            // Silently catch expected PyTorch errors
        }
        
        // Try chained unsqueeze operations if we have more data
        if (offset < Size) {
            try {
                int64_t rank2 = result.dim();
                int64_t range_size2 = 2 * rank2 + 2;
                int64_t dim2 = 0;
                if (range_size2 > 0) {
                    dim2 = static_cast<int64_t>(Data[offset++] % range_size2) - (rank2 + 1);
                }
                torch::Tensor result3 = result.unsqueeze(dim2);
                (void)result3;
            } catch (const c10::Error &) {
                // Silently catch expected PyTorch errors for invalid dimensions
            }
        }
        
        // Test unsqueeze with different dtypes
        if (offset < Size) {
            try {
                uint8_t dtype_selector = Data[offset++] % 4;
                torch::Tensor typed_tensor;
                switch (dtype_selector) {
                    case 0:
                        typed_tensor = input_tensor.to(torch::kFloat32);
                        break;
                    case 1:
                        typed_tensor = input_tensor.to(torch::kFloat64);
                        break;
                    case 2:
                        typed_tensor = input_tensor.to(torch::kInt32);
                        break;
                    default:
                        typed_tensor = input_tensor.to(torch::kInt64);
                        break;
                }
                torch::Tensor dtype_result = torch::unsqueeze(typed_tensor, 0);
                (void)dtype_result;
            } catch (const c10::Error &) {
                // Silently catch dtype conversion errors
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