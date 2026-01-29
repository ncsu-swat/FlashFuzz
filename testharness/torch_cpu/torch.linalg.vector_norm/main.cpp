#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For isfinite

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input is floating point (required for vector_norm)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Get parameters for vector_norm from the remaining data
        double ord = 2.0; // Default value
        if (offset + sizeof(double) <= Size) {
            double raw_ord;
            std::memcpy(&raw_ord, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Only use the value if it's finite, otherwise stick with default
            if (std::isfinite(raw_ord)) {
                ord = raw_ord;
            }
        }
        
        // Get dim parameter
        std::vector<int64_t> dim;
        if (offset < Size) {
            uint8_t dim_count = Data[offset++] % 4; // Limit to reasonable number of dimensions
            
            int64_t ndim = input.dim();
            for (uint8_t i = 0; i < dim_count && offset < Size; i++) {
                int8_t dim_value = static_cast<int8_t>(Data[offset++]);
                // Normalize to valid dimension range
                if (ndim > 0) {
                    dim_value = dim_value % ndim;
                } else {
                    dim_value = 0;
                }
                dim.push_back(dim_value);
            }
            
            // Remove duplicate dimensions to avoid errors
            std::sort(dim.begin(), dim.end());
            dim.erase(std::unique(dim.begin(), dim.end()), dim.end());
        }
        
        // Get keepdim parameter
        bool keepdim = false;
        if (offset < Size) {
            keepdim = Data[offset++] & 0x1;
        }
        
        // Get dtype parameter
        c10::optional<torch::ScalarType> dtype = c10::nullopt;
        if (offset + 1 < Size) {
            uint8_t use_dtype = Data[offset++] & 0x1;
            if (use_dtype) {
                torch::ScalarType parsed_dtype = fuzzer_utils::parseDataType(Data[offset++]);
                // Only use floating point dtypes for output
                if (parsed_dtype == torch::kFloat32 || 
                    parsed_dtype == torch::kFloat64 ||
                    parsed_dtype == torch::kFloat16 ||
                    parsed_dtype == torch::kBFloat16) {
                    dtype = parsed_dtype;
                }
            }
        }
        
        // Call vector_norm with different parameter combinations
        torch::Tensor result;
        
        try {
            // Try different combinations of parameters
            if (dim.empty() && !dtype.has_value()) {
                result = torch::linalg_vector_norm(input, ord, c10::nullopt, keepdim);
            } 
            else if (!dim.empty() && !dtype.has_value()) {
                result = torch::linalg_vector_norm(input, ord, dim, keepdim);
            }
            else if (dim.empty() && dtype.has_value()) {
                result = torch::linalg_vector_norm(input, ord, c10::nullopt, keepdim, dtype.value());
            }
            else {
                result = torch::linalg_vector_norm(input, ord, dim, keepdim, dtype.value());
            }
            
            // Verify result is valid by computing sum (works for any shape)
            if (result.defined() && result.numel() > 0) {
                auto sum = result.sum();
                (void)sum;
            }
        }
        catch (const c10::Error &e) {
            // Expected errors from invalid parameter combinations (e.g., invalid dims)
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