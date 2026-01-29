#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <set>            // For unique dimensions

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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor - frobenius_norm requires floating point tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert to float if not already floating point
        if (!input_tensor.is_floating_point()) {
            input_tensor = input_tensor.to(torch::kFloat32);
        }
        
        // Parse dim parameter if there's data left
        std::vector<int64_t> dim;
        if (offset + 1 < Size && input_tensor.dim() > 0) {
            uint8_t use_dim = Data[offset++];
            
            // Decide whether to use dim parameter
            if (use_dim % 2 == 1) {
                // Parse number of dimensions to use
                if (offset < Size) {
                    uint8_t num_dims = Data[offset++] % (input_tensor.dim() + 1);
                    
                    // Use a set to ensure unique dimensions
                    std::set<int64_t> dim_set;
                    for (uint8_t i = 0; i < num_dims && offset < Size; ++i) {
                        int64_t d = static_cast<int64_t>(Data[offset++]) % input_tensor.dim();
                        dim_set.insert(d);
                    }
                    
                    // Convert set to vector
                    for (int64_t d : dim_set) {
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
        
        try {
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
            
            // Access result to ensure computation is performed
            if (result.defined()) {
                // Use numel() and sum() instead of item() since result may not be scalar
                volatile auto numel = result.numel();
                if (numel == 1) {
                    volatile auto val = result.item<float>();
                    (void)val;
                } else {
                    // For non-scalar results, compute sum to ensure all values are accessed
                    volatile auto sum_val = result.sum().item<float>();
                    (void)sum_val;
                }
                (void)numel;
            }
        } catch (const c10::Error &e) {
            // Expected errors from invalid input combinations (e.g., empty tensor)
            // Silently catch these
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}