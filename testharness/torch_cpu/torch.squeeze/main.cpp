#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test squeeze without dimension
        torch::Tensor result1 = torch::squeeze(input_tensor);
        
        // Test squeeze with dimension if we have more data
        if (offset + 1 < Size) {
            // Get a dimension value from the input data
            int64_t dim = static_cast<int64_t>(Data[offset++]);
            
            // Allow negative dimensions for testing edge cases
            if (input_tensor.dim() > 0) {
                // Modulo to get a valid dimension index (including negative indices)
                dim = dim % (2 * input_tensor.dim()) - input_tensor.dim();
                
                // Apply squeeze with dimension
                torch::Tensor result2 = torch::squeeze(input_tensor, dim);
            }
        }
        
        // Test squeeze with dimension 0 explicitly
        if (input_tensor.dim() > 0) {
            torch::Tensor result3 = torch::squeeze(input_tensor, 0);
        }
        
        // Test squeeze with last dimension explicitly
        if (input_tensor.dim() > 0) {
            torch::Tensor result4 = torch::squeeze(input_tensor, input_tensor.dim() - 1);
        }
        
        // Test squeeze with out of bounds dimension (should throw exception)
        if (input_tensor.dim() > 0 && offset < Size) {
            try {
                int64_t out_of_bounds_dim = input_tensor.dim() + static_cast<int64_t>(Data[offset++]);
                torch::Tensor result5 = torch::squeeze(input_tensor, out_of_bounds_dim);
            } catch (const c10::Error &e) {
                // Expected exception for out of bounds dimension
            }
        }
        
        // Test squeeze on a tensor with no dimensions of size 1
        if (offset + 1 < Size) {
            std::vector<int64_t> non_one_dims;
            for (int i = 0; i < input_tensor.dim(); i++) {
                if (input_tensor.size(i) != 1) {
                    non_one_dims.push_back(i);
                }
            }
            
            if (!non_one_dims.empty()) {
                int64_t dim_idx = Data[offset++] % non_one_dims.size();
                int64_t non_one_dim = non_one_dims[dim_idx];
                torch::Tensor result6 = torch::squeeze(input_tensor, non_one_dim);
            }
        }
        
        // Test squeeze on a tensor with all dimensions of size 1
        if (offset < Size) {
            std::vector<int64_t> shape(Data[offset++] % 4 + 1, 1);
            torch::Tensor all_ones = torch::ones(shape);
            torch::Tensor result7 = torch::squeeze(all_ones);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
