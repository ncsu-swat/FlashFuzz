#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for meaningful fuzzing
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for diagonal operation
        int64_t offset_param = 0;
        int64_t dim1 = 0;
        int64_t dim2 = 1;
        
        // Parse offset parameter if we have more data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&offset_param, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse dim1 parameter if we have more data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim1, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Parse dim2 parameter if we have more data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim2, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Try different variants of the diagonal operation
        
        // Variant 1: Basic diagonal with default parameters
        try {
            torch::Tensor result1 = input_tensor.diagonal();
        } catch (const std::exception&) {
            // Catch and continue to next variant
        }
        
        // Variant 2: Diagonal with offset parameter
        try {
            torch::Tensor result2 = input_tensor.diagonal(offset_param);
        } catch (const std::exception&) {
            // Catch and continue to next variant
        }
        
        // Variant 3: Diagonal with offset and dim1 parameters
        try {
            torch::Tensor result3 = input_tensor.diagonal(offset_param, dim1);
        } catch (const std::exception&) {
            // Catch and continue to next variant
        }
        
        // Variant 4: Diagonal with all parameters
        try {
            torch::Tensor result4 = input_tensor.diagonal(offset_param, dim1, dim2);
        } catch (const std::exception&) {
            // Catch and continue
        }
        
        // Variant 5: Using functional API
        try {
            torch::Tensor result5 = torch::diagonal(input_tensor, offset_param, dim1, dim2);
        } catch (const std::exception&) {
            // Catch and continue
        }
        
        // Variant 6: Using negative dimensions
        try {
            int64_t neg_dim1 = -dim1;
            int64_t neg_dim2 = -dim2;
            torch::Tensor result6 = input_tensor.diagonal(offset_param, neg_dim1, neg_dim2);
        } catch (const std::exception&) {
            // Catch and continue
        }
        
        // Variant 7: Using extreme offset values
        try {
            int64_t extreme_offset = (offset_param % 2 == 0) ? 
                                     std::numeric_limits<int64_t>::max() / 2 : 
                                     std::numeric_limits<int64_t>::min() / 2;
            torch::Tensor result7 = input_tensor.diagonal(extreme_offset);
        } catch (const std::exception&) {
            // Catch and continue
        }
        
        // Variant 8: Using same dimension for dim1 and dim2
        try {
            torch::Tensor result8 = input_tensor.diagonal(offset_param, dim1, dim1);
        } catch (const std::exception&) {
            // Catch and continue
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}