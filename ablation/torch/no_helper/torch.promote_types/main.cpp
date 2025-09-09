#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>
#include <vector>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least 2 bytes for two dtype indices
        if (Size < 2) {
            return 0;
        }

        // Define all available dtypes for testing
        std::vector<torch::ScalarType> scalar_types = {
            torch::kUInt8,
            torch::kInt8,
            torch::kInt16,
            torch::kInt32,
            torch::kInt64,
            torch::kFloat16,
            torch::kFloat32,
            torch::kFloat64,
            torch::kComplexFloat32,
            torch::kComplexFloat64,
            torch::kBool
        };

        // Extract indices for two dtypes from fuzzer input
        uint8_t type1_idx = Data[offset++] % scalar_types.size();
        uint8_t type2_idx = Data[offset++] % scalar_types.size();

        torch::ScalarType type1 = scalar_types[type1_idx];
        torch::ScalarType type2 = scalar_types[type2_idx];

        // Test torch::promote_types with the selected types
        torch::ScalarType promoted_type = torch::promote_types(type1, type2);

        // Additional validation: ensure the promoted type is valid
        // The promoted type should be at least as capable as both input types
        torch::Tensor tensor1 = torch::zeros({1}, torch::dtype(type1));
        torch::Tensor tensor2 = torch::zeros({1}, torch::dtype(type2));
        torch::Tensor promoted_tensor = torch::zeros({1}, torch::dtype(promoted_type));

        // Test that we can convert both input types to the promoted type
        tensor1.to(promoted_type);
        tensor2.to(promoted_type);

        // Test edge cases with same types
        if (offset < Size) {
            torch::ScalarType same_type = scalar_types[Data[offset++] % scalar_types.size()];
            torch::ScalarType self_promoted = torch::promote_types(same_type, same_type);
            
            // Self-promotion should return the same type
            if (self_promoted != same_type) {
                std::cout << "Self-promotion inconsistency detected" << std::endl;
            }
        }

        // Test commutativity: promote_types(A, B) should equal promote_types(B, A)
        torch::ScalarType promoted_ab = torch::promote_types(type1, type2);
        torch::ScalarType promoted_ba = torch::promote_types(type2, type1);
        
        if (promoted_ab != promoted_ba) {
            std::cout << "Commutativity violation detected" << std::endl;
        }

        // Test with additional type combinations if we have more data
        while (offset + 1 < Size) {
            uint8_t extra_type1_idx = Data[offset++] % scalar_types.size();
            uint8_t extra_type2_idx = Data[offset++] % scalar_types.size();
            
            torch::ScalarType extra_type1 = scalar_types[extra_type1_idx];
            torch::ScalarType extra_type2 = scalar_types[extra_type2_idx];
            
            torch::ScalarType extra_promoted = torch::promote_types(extra_type1, extra_type2);
            
            // Create tensors to verify the promotion is valid
            torch::zeros({1}, torch::dtype(extra_type1)).to(extra_promoted);
            torch::zeros({1}, torch::dtype(extra_type2)).to(extra_promoted);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}