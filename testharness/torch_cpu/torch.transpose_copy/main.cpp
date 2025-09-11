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
        
        // Extract dimensions for transpose
        int64_t dim0 = 0;
        int64_t dim1 = 0;
        
        // Get dimensions to transpose if we have enough data
        if (offset + sizeof(int64_t) * 2 <= Size) {
            std::memcpy(&dim0, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            std::memcpy(&dim1, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Get tensor rank
        int64_t tensor_rank = input_tensor.dim();
        
        // If tensor has at least 2 dimensions, ensure dims are within valid range
        if (tensor_rank >= 2) {
            // Map dimensions to valid range for the tensor
            dim0 = std::abs(dim0) % tensor_rank;
            dim1 = std::abs(dim1) % tensor_rank;
            
            // Apply transpose_copy operation
            torch::Tensor output = torch::transpose_copy(input_tensor, dim0, dim1);
            
            // Verify the output is valid
            if (output.defined()) {
                // Access elements to ensure no segfaults
                if (output.numel() > 0) {
                    output.item();
                }
            }
        } else if (tensor_rank == 1) {
            // For 1D tensors, try transpose with dim0=0, dim1=0
            torch::Tensor output = torch::transpose_copy(input_tensor, 0, 0);
            
            // Verify the output is valid
            if (output.defined() && output.numel() > 0) {
                output.item();
            }
        } else if (tensor_rank == 0) {
            // For 0D tensors, try transpose with various dimensions
            // This tests error handling for scalar tensors
            try {
                torch::Tensor output = torch::transpose_copy(input_tensor, 0, 0);
                if (output.defined() && output.numel() > 0) {
                    output.item();
                }
            } catch (...) {
                // Expected to fail for scalar tensors
            }
            
            try {
                torch::Tensor output = torch::transpose_copy(input_tensor, dim0, dim1);
                if (output.defined() && output.numel() > 0) {
                    output.item();
                }
            } catch (...) {
                // Expected to fail for scalar tensors
            }
        }
        
        // Test with negative dimensions
        if (tensor_rank >= 2) {
            try {
                // Negative dimensions should wrap around
                torch::Tensor output = torch::transpose_copy(input_tensor, -1, -2);
                if (output.defined() && output.numel() > 0) {
                    output.item();
                }
            } catch (...) {
                // May fail depending on implementation
            }
        }
        
        // Test with out-of-bounds dimensions
        try {
            torch::Tensor output = torch::transpose_copy(input_tensor, tensor_rank, tensor_rank + 1);
            if (output.defined() && output.numel() > 0) {
                output.item();
            }
        } catch (...) {
            // Expected to fail for out-of-bounds dimensions
        }
        
        // Test with same dimension twice
        if (tensor_rank >= 1) {
            try {
                int64_t same_dim = tensor_rank > 0 ? (std::abs(dim0) % tensor_rank) : 0;
                torch::Tensor output = torch::transpose_copy(input_tensor, same_dim, same_dim);
                if (output.defined() && output.numel() > 0) {
                    output.item();
                }
            } catch (...) {
                // May or may not fail depending on implementation
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
