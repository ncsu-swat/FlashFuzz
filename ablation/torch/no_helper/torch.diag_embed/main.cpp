#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least 16 bytes for basic parameters
        if (Size < 16) {
            return 0;
        }

        // Parse input tensor configuration
        auto tensor_config = parseTensorConfig(Data, Size, offset);
        if (!tensor_config.has_value()) {
            return 0;
        }

        // Create input tensor - must be at least 1-dimensional
        auto input = createTensor(tensor_config.value());
        if (input.dim() == 0) {
            // Make it at least 1D
            input = input.unsqueeze(0);
        }

        // Parse offset parameter (int)
        int diag_offset = 0;
        if (offset + sizeof(int) <= Size) {
            diag_offset = *reinterpret_cast<const int*>(Data + offset);
            offset += sizeof(int);
            // Clamp offset to reasonable range to avoid excessive memory usage
            diag_offset = std::max(-100, std::min(100, diag_offset));
        }

        // Parse dim1 parameter (int)
        int dim1 = -2;
        if (offset + sizeof(int) <= Size) {
            dim1 = *reinterpret_cast<const int*>(Data + offset);
            offset += sizeof(int);
            // Normalize dimension to valid range
            int ndim = input.dim() + 1; // diag_embed adds one dimension
            if (dim1 < 0) {
                dim1 = ndim + dim1;
            }
            dim1 = std::max(0, std::min(ndim - 1, dim1));
        }

        // Parse dim2 parameter (int)
        int dim2 = -1;
        if (offset + sizeof(int) <= Size) {
            dim2 = *reinterpret_cast<const int*>(Data + offset);
            offset += sizeof(int);
            // Normalize dimension to valid range
            int ndim = input.dim() + 1; // diag_embed adds one dimension
            if (dim2 < 0) {
                dim2 = ndim + dim2;
            }
            dim2 = std::max(0, std::min(ndim - 1, dim2));
        }

        // Ensure dim1 and dim2 are different
        if (dim1 == dim2) {
            dim2 = (dim1 + 1) % (input.dim() + 1);
        }

        // Test basic diag_embed operation
        auto result1 = torch::diag_embed(input);
        
        // Test with offset parameter
        auto result2 = torch::diag_embed(input, diag_offset);
        
        // Test with all parameters
        auto result3 = torch::diag_embed(input, diag_offset, dim1, dim2);

        // Test edge cases with different input shapes
        if (input.numel() > 0) {
            // Test with flattened input
            auto flat_input = input.flatten();
            auto result4 = torch::diag_embed(flat_input);
            
            // Test with reshaped input if possible
            if (input.numel() >= 4) {
                auto reshaped = input.view({2, -1});
                auto result5 = torch::diag_embed(reshaped);
            }
        }

        // Test with different offsets
        for (int test_offset : {-2, -1, 0, 1, 2}) {
            try {
                auto result_offset = torch::diag_embed(input, test_offset);
            } catch (...) {
                // Some offsets might be invalid, continue testing
            }
        }

        // Test with different dimension combinations
        int max_dim = input.dim();
        for (int d1 = 0; d1 <= max_dim; d1++) {
            for (int d2 = 0; d2 <= max_dim; d2++) {
                if (d1 != d2) {
                    try {
                        auto result_dims = torch::diag_embed(input, 0, d1, d2);
                    } catch (...) {
                        // Some dimension combinations might be invalid
                    }
                }
            }
        }

        // Test with negative dimensions
        try {
            auto result_neg = torch::diag_embed(input, 0, -2, -1);
        } catch (...) {
            // Negative dims might be invalid for some tensor shapes
        }

        // Test consistency: applying diagonal to diag_embed result should give back input
        try {
            auto embedded = torch::diag_embed(input, diag_offset, dim1, dim2);
            auto extracted = torch::diagonal(embedded, diag_offset, dim1, dim2);
            // Note: shapes might differ due to broadcasting, so we don't assert equality
        } catch (...) {
            // This test might fail for some parameter combinations
        }

        // Test with different data types if input allows
        if (input.dtype() == torch::kFloat32) {
            try {
                auto int_input = input.to(torch::kInt32);
                auto result_int = torch::diag_embed(int_input);
            } catch (...) {
                // Type conversion might fail
            }
        }

        // Test memory efficiency with large tensors
        if (input.numel() > 1000) {
            // For large inputs, test with small offsets only
            auto result_large = torch::diag_embed(input, 0);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}