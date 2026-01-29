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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor for bincount
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // bincount requires 1-D tensor with non-negative integers
        // Flatten to 1-D
        input = input.flatten();
        
        // Convert to int64 if not integer type
        if (!input.is_floating_point()) {
            input = input.to(torch::kInt64);
        } else {
            // For floating point, take absolute value and convert
            input = input.abs().to(torch::kInt64);
        }
        
        // Ensure non-negative values (bincount requirement)
        input = input.abs();
        
        // Limit max value to avoid excessive memory allocation
        // bincount output size is max(input) + 1
        input = input.remainder(1000);
        
        // Create optional weights tensor
        bool use_weights = false;
        torch::Tensor weights;
        
        // Use remaining data to decide if we should use weights
        if (offset < Size) {
            use_weights = (Data[offset++] % 2 == 0);
            
            if (use_weights && offset < Size) {
                weights = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Weights must be same length as input (1-D)
                weights = weights.flatten();
                
                // Resize weights to match input length
                int64_t input_len = input.size(0);
                int64_t weights_len = weights.size(0);
                
                if (weights_len > input_len) {
                    weights = weights.slice(0, 0, input_len);
                } else if (weights_len < input_len) {
                    // Repeat weights to match input length
                    int64_t repeats = (input_len + weights_len - 1) / weights_len;
                    weights = weights.repeat({repeats}).slice(0, 0, input_len);
                }
                
                // Convert weights to float if needed (bincount requires float/double weights)
                if (weights.dtype() != torch::kFloat32 && weights.dtype() != torch::kFloat64) {
                    weights = weights.to(torch::kFloat32);
                }
            }
        }
        
        // Get minlength parameter (optional)
        int64_t minlength = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&minlength, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure minlength is non-negative and not too large
            minlength = std::abs(minlength) % 1000;
        }
        
        // Call bincount with different parameter combinations
        torch::Tensor result;
        
        try {
            if (use_weights && weights.defined()) {
                result = torch::bincount(input, weights, minlength);
            } else {
                result = torch::bincount(input, {}, minlength);
            }
            
            // Access result to ensure computation is performed
            auto result_size = result.sizes();
            auto result_sum = result.sum().item<int64_t>();
            (void)result_sum;
        } catch (const c10::Error &e) {
            // Expected errors from invalid inputs (shape mismatches, etc.)
            // Silently ignore these
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}