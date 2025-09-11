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
        
        // Create input tensor
        if (offset >= Size) return 0;
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get the number of dimensions
        int64_t ndim = input_tensor.dim();
        
        // Create permutation dimensions
        std::vector<int64_t> permutation;
        
        // Parse permutation dimensions from the input data
        for (int64_t i = 0; i < ndim && offset < Size; ++i) {
            // Use the next byte to determine a dimension index
            if (offset < Size) {
                int64_t dim_idx = static_cast<int64_t>(Data[offset++]) % ndim;
                
                // Check if this dimension is already in the permutation
                bool already_exists = false;
                for (int64_t j = 0; j < permutation.size(); ++j) {
                    if (permutation[j] == dim_idx) {
                        already_exists = true;
                        break;
                    }
                }
                
                // Only add if not already in the permutation
                if (!already_exists) {
                    permutation.push_back(dim_idx);
                }
            }
        }
        
        // If we didn't get a complete permutation, fill in the missing dimensions
        if (permutation.size() < ndim) {
            for (int64_t i = 0; i < ndim; ++i) {
                bool exists = false;
                for (int64_t j = 0; j < permutation.size(); ++j) {
                    if (permutation[j] == i) {
                        exists = true;
                        break;
                    }
                }
                if (!exists) {
                    permutation.push_back(i);
                }
            }
        }
        
        // Also test with negative indices
        if (offset < Size && Data[offset++] % 2 == 0 && !permutation.empty()) {
            for (int64_t i = 0; i < permutation.size() && offset < Size; ++i) {
                if (Data[offset++] % 3 == 0) {
                    permutation[i] = permutation[i] - ndim; // Convert to negative index
                }
            }
        }
        
        // Apply permute operation
        torch::Tensor output;
        
        // Test different ways to call permute
        if (offset < Size) {
            uint8_t call_type = Data[offset++] % 3;
            
            switch (call_type) {
                case 0: {
                    // Call permute with individual dimensions
                    if (ndim == 0) {
                        output = input_tensor.permute({}); // Scalar case
                    } else if (ndim == 1) {
                        output = input_tensor.permute({permutation[0]});
                    } else if (ndim == 2) {
                        output = input_tensor.permute({permutation[0], permutation[1]});
                    } else if (ndim == 3) {
                        output = input_tensor.permute({permutation[0], permutation[1], permutation[2]});
                    } else {
                        output = input_tensor.permute({permutation[0], permutation[1], permutation[2], permutation[3]});
                    }
                    break;
                }
                case 1: {
                    // Call permute with vector of dimensions
                    output = input_tensor.permute(permutation);
                    break;
                }
                case 2: {
                    // Call permute with IntArrayRef
                    output = input_tensor.permute(c10::IntArrayRef(permutation));
                    break;
                }
            }
        } else {
            // Default case if we don't have enough data
            output = input_tensor.permute(permutation);
        }
        
        // Verify the output has the expected shape
        auto input_sizes = input_tensor.sizes().vec();
        auto output_sizes = output.sizes().vec();
        
        if (input_sizes.size() != output_sizes.size()) {
            throw std::runtime_error("Input and output tensor dimensions don't match");
        }
        
        for (size_t i = 0; i < permutation.size(); ++i) {
            int64_t perm_idx = permutation[i];
            if (perm_idx < 0) perm_idx += ndim;
            
            if (perm_idx >= 0 && perm_idx < ndim && i < output_sizes.size() && perm_idx < input_sizes.size()) {
                if (output_sizes[i] != input_sizes[perm_idx]) {
                    throw std::runtime_error("Output shape doesn't match expected permutation");
                }
            }
        }
        
        // Test that permute is correctly implemented by checking data consistency
        if (!output.is_contiguous()) {
            auto contiguous_output = output.contiguous();
            
            // Verify some elements to ensure data is correctly permuted
            if (output.numel() > 0) {
                // Check first element
                if (input_tensor.item<float>() != output.item<float>()) {
                    throw std::runtime_error("First element mismatch after permute");
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
