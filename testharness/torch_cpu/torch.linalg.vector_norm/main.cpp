#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get parameters for vector_norm from the remaining data
        double ord = 2.0; // Default value
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&ord, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Get dim parameter
        std::vector<int64_t> dim;
        if (offset < Size) {
            uint8_t dim_count = Data[offset++] % 4; // Limit to reasonable number of dimensions
            
            for (uint8_t i = 0; i < dim_count && offset + sizeof(int64_t) <= Size; i++) {
                int64_t dim_value;
                std::memcpy(&dim_value, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Allow negative dimensions for testing edge cases
                dim.push_back(dim_value);
            }
        }
        
        // Get keepdim parameter
        bool keepdim = false;
        if (offset < Size) {
            keepdim = Data[offset++] & 0x1;
        }
        
        // Get dtype parameter
        c10::optional<torch::ScalarType> dtype = c10::nullopt;
        if (offset < Size) {
            uint8_t use_dtype = Data[offset++] & 0x1;
            if (use_dtype) {
                dtype = fuzzer_utils::parseDataType(Data[offset++]);
            }
        }
        
        // Call vector_norm with different parameter combinations
        torch::Tensor result;
        
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
        
        // Try to access result to ensure computation is performed
        if (result.defined()) {
            auto item = result.item();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}