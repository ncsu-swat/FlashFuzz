#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <optional>       // For std::optional spacing handling
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse spacing parameters if we have more data
        std::vector<torch::Tensor> spacing;
        if (offset + 1 < Size) {
            uint8_t num_spacing = Data[offset++] % 3; // 0, 1, or 2 spacing tensors
            
            for (uint8_t i = 0; i < num_spacing && offset < Size; ++i) {
                if (Size - offset > 2) {
                    torch::Tensor spacing_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                    spacing.push_back(spacing_tensor);
                }
            }
        }
        
        // Parse edge_order parameter
        int64_t edge_order = 1;
        if (offset < Size) {
            edge_order = static_cast<int64_t>(Data[offset++]) % 3; // 0, 1, or 2
        }
        
        // Parse dim parameter
        std::vector<int64_t> dim;
        if (offset + 1 < Size) {
            uint8_t num_dims = Data[offset++] % 5; // 0 to 4 dimensions
            
            for (uint8_t i = 0; i < num_dims && offset < Size; ++i) {
                if (offset < Size) {
                    int64_t d = static_cast<int64_t>(Data[offset++]) % (input.dim() + 2) - 1; // Allow -1 and out-of-bounds
                    dim.push_back(d);
                }
            }
        }
        
        // Call torch::gradient with different parameter combinations
        std::vector<torch::Tensor> result;
        
        // Case 1: Basic gradient call
        try {
            result = torch::gradient(input);
        } catch (const std::exception&) {
            // Exception caught, continue with other cases
        }
        
        // Case 2: With spacing
        if (!spacing.empty()) {
            try {
                result = torch::gradient(input, spacing);
            } catch (const std::exception&) {
                // Exception caught, continue with other cases
            }
        }
        
        // Case 3: With edge_order
        try {
            result = torch::gradient(input, torch::nullopt, torch::nullopt, edge_order);
        } catch (const std::exception&) {
            // Exception caught, continue with other cases
        }
        
        // Case 4: With dim
        if (!dim.empty()) {
            try {
                result = torch::gradient(input, dim, edge_order);
            } catch (const std::exception&) {
                // Exception caught, continue with other cases
            }
        }
        
        // Case 5: With all parameters
        if (!spacing.empty() && !dim.empty()) {
            try {
                result = torch::gradient(input, spacing, dim, edge_order);
            } catch (const std::exception&) {
                // Exception caught, continue with other cases
            }
        }
        
        // Case 6: With scalar spacing
        if (offset < Size) {
            double scalar_spacing = static_cast<double>(Data[offset++]) / 10.0;
            try {
                std::optional<torch::Scalar> spacing_scalar = torch::Scalar(scalar_spacing);
                result = torch::gradient(input, spacing_scalar);
            } catch (const std::exception&) {
                // Exception caught, continue with other cases
            }
        }
        
        // Case 7: With scalar spacing and edge_order
        if (offset < Size) {
            double scalar_spacing = static_cast<double>(Data[offset++]) / 10.0;
            try {
                std::optional<torch::Scalar> spacing_scalar = torch::Scalar(scalar_spacing);
                result = torch::gradient(input, spacing_scalar, std::nullopt, edge_order);
            } catch (const std::exception&) {
                // Exception caught, continue with other cases
            }
        }
        
        // Case 8: With scalar spacing, edge_order, and dim
        if (offset < Size && !dim.empty()) {
            double scalar_spacing = static_cast<double>(Data[offset++]) / 10.0;
            try {
                std::vector<torch::Scalar> spacing_scalar{torch::Scalar(scalar_spacing)};
                result = torch::gradient(input, spacing_scalar, dim, edge_order);
            } catch (const std::exception&) {
                // Exception caught, continue with other cases
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
