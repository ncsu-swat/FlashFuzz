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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse dim parameter if there's data left
        std::vector<int64_t> dim;
        if (offset + 1 < Size) {
            uint8_t use_dim = Data[offset++];
            
            // Decide whether to use dim parameter
            if (use_dim % 2 == 1 && input_tensor.dim() > 0) {
                // Parse number of dimensions to use
                if (offset < Size) {
                    uint8_t num_dims = Data[offset++] % (input_tensor.dim() + 1);
                    
                    // Parse each dimension
                    for (uint8_t i = 0; i < num_dims && offset < Size; ++i) {
                        int64_t d = static_cast<int64_t>(Data[offset++]) % input_tensor.dim();
                        dim.push_back(d);
                    }
                }
            }
        }
        
        // Parse keepdim parameter if there's data left
        bool keepdim = false;
        if (offset < Size) {
            keepdim = Data[offset++] % 2 == 1;
        }
        
        // Apply frobenius_norm operation with different parameter combinations
        torch::Tensor result;
        
        if (dim.empty()) {
            // Case 1: No dim specified - use all dimensions
            std::vector<int64_t> all_dims;
            for (int64_t i = 0; i < input_tensor.dim(); ++i) {
                all_dims.push_back(i);
            }
            result = torch::frobenius_norm(input_tensor, all_dims, keepdim);
        } else {
            // Case 2: With dim and keepdim
            result = torch::frobenius_norm(input_tensor, dim, keepdim);
        }
        
        // Try to access result to ensure computation is performed
        if (result.defined()) {
            auto item = result.item();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
