#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least 8 bytes for basic parameters
        if (Size < 8) {
            return 0;
        }

        // Parse tensor dimensions and properties
        auto dims = parse_tensor_dims(Data, Size, offset, 4); // Max 4 dimensions
        if (dims.empty()) {
            return 0;
        }

        // Parse data type
        auto dtype = parse_dtype(Data, Size, offset);
        
        // Parse offset parameter for diagflat
        int diagflat_offset = 0;
        if (offset + sizeof(int32_t) <= Size) {
            diagflat_offset = static_cast<int>(parse_int32(Data, Size, offset));
            // Clamp offset to reasonable range to avoid excessive memory usage
            diagflat_offset = std::max(-100, std::min(100, diagflat_offset));
        }

        // Create input tensor with parsed dimensions
        torch::Tensor input;
        
        // Test different tensor shapes and edge cases
        if (dims.size() == 1) {
            // 1D tensor case - creates square diagonal matrix
            int64_t size = std::max(1L, std::min(100L, dims[0])); // Reasonable size limits
            input = create_tensor({size}, dtype, Data, Size, offset);
        } else {
            // Multi-dimensional tensor case - flattened then used as diagonal
            // Limit total elements to prevent excessive memory usage
            int64_t total_elements = 1;
            std::vector<int64_t> clamped_dims;
            for (auto dim : dims) {
                int64_t clamped_dim = std::max(1L, std::min(20L, dim));
                clamped_dims.push_back(clamped_dim);
                total_elements *= clamped_dim;
                if (total_elements > 1000) break; // Prevent excessive memory usage
            }
            
            if (total_elements <= 1000) {
                input = create_tensor(clamped_dims, dtype, Data, Size, offset);
            } else {
                // Fallback to smaller tensor
                input = create_tensor({5, 5}, dtype, Data, Size, offset);
            }
        }

        // Test torch::diagflat with different scenarios
        
        // Basic diagflat call with default offset (0)
        auto result1 = torch::diagflat(input);
        
        // Test with positive offset
        auto result2 = torch::diagflat(input, diagflat_offset);
        
        // Test with negative offset
        auto result3 = torch::diagflat(input, -std::abs(diagflat_offset));
        
        // Test edge cases
        
        // Empty tensor case
        if (input.numel() > 0) {
            auto empty_input = torch::empty({0}, input.dtype());
            auto empty_result = torch::diagflat(empty_input);
        }
        
        // Single element tensor
        auto single_elem = torch::ones({1}, input.dtype());
        auto single_result = torch::diagflat(single_elem, diagflat_offset);
        
        // Test with different data types if original tensor allows
        if (input.dtype() == torch::kFloat32) {
            auto int_input = input.to(torch::kInt32);
            auto int_result = torch::diagflat(int_input, diagflat_offset);
        }
        
        // Verify basic properties of results
        if (result1.defined()) {
            // Result should be 2D
            if (result1.dim() != 2) {
                throw std::runtime_error("diagflat result should be 2D");
            }
            
            // Result should be square
            if (result1.size(0) != result1.size(1)) {
                throw std::runtime_error("diagflat result should be square");
            }
        }
        
        // Test with very large offset (should still work but create larger matrix)
        if (input.numel() > 0 && input.numel() < 10) {
            auto large_offset_result = torch::diagflat(input, 50);
            auto neg_large_offset_result = torch::diagflat(input, -50);
        }
        
        // Test contiguous vs non-contiguous input
        if (input.dim() > 1) {
            auto transposed = input.transpose(0, -1);
            auto transposed_result = torch::diagflat(transposed, diagflat_offset);
        }
        
        // Force evaluation of results to catch any lazy evaluation issues
        if (result1.defined()) result1.sum();
        if (result2.defined()) result2.sum();
        if (result3.defined()) result3.sum();

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}