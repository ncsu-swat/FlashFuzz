#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Parse tensor dimensions and data type
        auto dims = parse_tensor_dims(Data, Size, offset);
        if (dims.empty()) return 0;
        
        auto dtype = parse_dtype(Data, Size, offset);
        
        // Create input tensor with parsed dimensions and dtype
        auto input = create_tensor(Data, Size, offset, dims, dtype);
        if (!input.defined()) return 0;

        // Parse optional parameters for torch.unique
        bool sorted = parse_bool(Data, Size, offset);
        bool return_inverse = parse_bool(Data, Size, offset);
        bool return_counts = parse_bool(Data, Size, offset);
        
        // Parse optional dim parameter (-1 means no dim specified)
        int64_t dim = -1;
        bool use_dim = parse_bool(Data, Size, offset);
        if (use_dim && input.dim() > 0) {
            dim = parse_int64_range(Data, Size, offset, -input.dim(), input.dim() - 1);
        }

        // Test torch.unique with different parameter combinations
        
        // Case 1: Basic unique without dim
        if (!use_dim) {
            auto result1 = torch::unique(input, sorted, return_inverse, return_counts);
            // Access all returned values to ensure they're computed
            auto unique_vals = std::get<0>(result1);
            if (return_inverse) {
                auto inverse_indices = std::get<1>(result1);
            }
            if (return_counts) {
                auto counts = std::get<2>(result1);
            }
        }
        
        // Case 2: Unique with dim parameter
        if (use_dim && input.dim() > 0) {
            auto result2 = torch::unique_dim(input, dim, sorted, return_inverse, return_counts);
            auto unique_vals = std::get<0>(result2);
            if (return_inverse) {
                auto inverse_indices = std::get<1>(result2);
            }
            if (return_counts) {
                auto counts = std::get<2>(result2);
            }
        }

        // Case 3: Test with consecutive unique
        auto consecutive_result = torch::unique_consecutive(input, return_inverse, return_counts, use_dim ? c10::optional<int64_t>(dim) : c10::nullopt);
        auto consecutive_unique = std::get<0>(consecutive_result);
        if (return_inverse) {
            auto consecutive_inverse = std::get<1>(consecutive_result);
        }
        if (return_counts) {
            auto consecutive_counts = std::get<2>(consecutive_result);
        }

        // Test edge cases with different tensor shapes
        if (input.numel() > 0) {
            // Test with flattened tensor
            auto flattened = input.flatten();
            auto flat_result = torch::unique(flattened, sorted, return_inverse, return_counts);
            
            // Test with reshaped tensor (if possible)
            if (input.numel() >= 4) {
                try {
                    auto reshaped = input.view({-1, 2});
                    auto reshape_result = torch::unique_dim(reshaped, 0, sorted, return_inverse, return_counts);
                } catch (...) {
                    // Ignore reshape failures
                }
            }
        }

        // Test with cloned tensor to ensure no memory issues
        auto cloned = input.clone();
        auto clone_result = torch::unique(cloned, sorted, return_inverse, return_counts);

        // Test with detached tensor
        auto detached = input.detach();
        auto detach_result = torch::unique(detached, sorted, return_inverse, return_counts);

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}