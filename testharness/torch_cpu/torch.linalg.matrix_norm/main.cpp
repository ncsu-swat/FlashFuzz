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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor has at least 2 dimensions for matrix_norm
        if (input.dim() < 2) {
            // Add dimensions if needed
            if (input.dim() == 0) {
                input = input.unsqueeze(0).unsqueeze(0);
            } else if (input.dim() == 1) {
                input = input.unsqueeze(0);
            }
        }
        
        // Get remaining bytes for norm parameters
        if (offset >= Size) {
            offset = 0; // Reset if we've consumed all data
        }
        
        // Parse norm type (1, 2, inf, -inf, 'fro', 'nuc')
        uint8_t norm_selector = (offset < Size) ? Data[offset++] : 0;
        std::vector<std::string> norm_types = {"fro", "nuc"};
        std::vector<double> norm_values = {1.0, 2.0, INFINITY, -INFINITY};
        
        // Choose norm type based on selector
        c10::optional<torch::Scalar> norm;
        std::string norm_str;
        if (norm_selector % 2 == 0) {
            // Use string norm type
            norm_str = norm_types[norm_selector % norm_types.size()];
            norm = c10::nullopt; // Will use string version
        } else {
            // Use numeric norm type
            double norm_val = norm_values[(norm_selector / 2) % norm_values.size()];
            norm = torch::Scalar(norm_val);
        }
        
        // Parse dim parameter
        std::vector<int64_t> dim;
        if (offset < Size) {
            uint8_t dim_selector = Data[offset++];
            if (dim_selector % 3 == 0) {
                // Use (-2, -1) dims
                dim = {-2, -1};
            } else if (dim_selector % 3 == 1) {
                // Use last two dimensions
                int64_t last_dim = input.dim() - 1;
                int64_t second_last_dim = std::max(0L, last_dim - 1);
                dim = {second_last_dim, last_dim};
            } else {
                // Use first two dimensions
                dim = {0, std::min(1L, static_cast<int64_t>(input.dim() - 1))};
            }
        } else {
            // Default to last two dimensions
            int64_t last_dim = input.dim() - 1;
            int64_t second_last_dim = std::max(0L, last_dim - 1);
            dim = {second_last_dim, last_dim};
        }
        
        // Parse keepdim parameter
        bool keepdim = false;
        if (offset < Size) {
            keepdim = Data[offset++] % 2 == 0;
        }
        
        // Parse dtype parameter
        c10::optional<torch::ScalarType> dtype = c10::nullopt;
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            if (dtype_selector % 3 != 0) { // 2/3 chance to use a dtype
                dtype = fuzzer_utils::parseDataType(dtype_selector);
            }
        }
        
        // Apply matrix_norm operation
        torch::Tensor result;
        
        // Try different combinations of parameters
        if (offset < Size) {
            uint8_t param_selector = Data[offset++];
            
            switch (param_selector % 4) {
                case 0:
                    // Basic call with norm only
                    if (norm.has_value()) {
                        result = torch::matrix_norm(input, norm.value());
                    } else {
                        result = torch::matrix_norm(input, norm_str);
                    }
                    break;
                case 1:
                    // With norm and dim
                    if (norm.has_value()) {
                        result = torch::matrix_norm(input, norm.value(), dim);
                    } else {
                        result = torch::matrix_norm(input, norm_str, dim);
                    }
                    break;
                case 2:
                    // With norm, dim, and keepdim
                    if (norm.has_value()) {
                        result = torch::matrix_norm(input, norm.value(), dim, keepdim);
                    } else {
                        result = torch::matrix_norm(input, norm_str, dim, keepdim);
                    }
                    break;
                case 3:
                    // With all parameters
                    if (norm.has_value()) {
                        result = torch::matrix_norm(input, norm.value(), dim, keepdim, dtype);
                    } else {
                        result = torch::matrix_norm(input, norm_str, dim, keepdim, dtype);
                    }
                    break;
            }
        } else {
            // Default call if we've consumed all data
            if (norm.has_value()) {
                result = torch::matrix_norm(input, norm.value());
            } else {
                result = torch::matrix_norm(input, norm_str);
            }
        }
        
        // Verify result is not empty
        if (result.numel() == 0) {
            throw std::runtime_error("Result tensor is empty");
        }
    }
    catch (const std::exception &e)
    {
        return 0; // keep the input
    }
    return 0; // keep the input
}
