#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <algorithm>      // For std::find

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
        
        // Need at least a few bytes to create a tensor and specify dimensions
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract dim parameter from the remaining data
        int64_t dim = 0;
        bool keepdim = false;
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Extract keepdim parameter
        if (offset < Size) {
            keepdim = static_cast<bool>(Data[offset] & 0x1);
            offset++;
        }
        
        // Apply logsumexp operation
        torch::Tensor result;
        
        // If tensor has no dimensions (scalar), apply without dim parameter
        if (input.dim() == 0) {
            // For 0-d tensors, we need to use an empty dim vector
            std::vector<int64_t> empty_dims;
            try {
                result = torch::logsumexp(input, empty_dims, keepdim);
            } catch (const std::exception &) {
                // Expected for some edge cases
            }
        } else {
            // Normalize dim to valid range [-input.dim(), input.dim()-1]
            int64_t ndim = input.dim();
            dim = ((dim % ndim) + ndim) % ndim;  // Ensure positive valid index
            
            // Apply logsumexp with specified dim
            result = torch::logsumexp(input, dim, keepdim);
        }
        
        // Try with multiple dimensions if tensor has enough dimensions
        if (input.dim() >= 2 && offset < Size) {
            try {
                std::vector<int64_t> dims;
                int num_dims = (Data[offset++] % (input.dim() - 1)) + 1;  // At least 1, at most input.dim()-1
                
                for (int i = 0; i < num_dims && offset < Size; i++) {
                    int64_t d = static_cast<int64_t>(Data[offset++]) % input.dim();
                    if (std::find(dims.begin(), dims.end(), d) == dims.end()) {
                        dims.push_back(d);
                    }
                }
                
                if (!dims.empty()) {
                    torch::Tensor result_multi = torch::logsumexp(input, dims, keepdim);
                }
            } catch (const std::exception &) {
                // Expected for some dimension combinations
            }
        }
        
        // Test with negative dimension index
        if (input.dim() > 0 && offset < Size) {
            try {
                int64_t neg_dim = -(static_cast<int64_t>(Data[offset++] % input.dim()) + 1);
                torch::Tensor result_neg = torch::logsumexp(input, neg_dim, !keepdim);
            } catch (const std::exception &) {
                // Expected for some cases
            }
        }
        
        // Test with IntArrayRef containing all dimensions (reduction to scalar)
        if (input.dim() > 0) {
            try {
                std::vector<int64_t> all_dims;
                for (int64_t i = 0; i < input.dim(); i++) {
                    all_dims.push_back(i);
                }
                torch::Tensor result_all = torch::logsumexp(input, all_dims, keepdim);
            } catch (const std::exception &) {
                // Expected for some cases
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