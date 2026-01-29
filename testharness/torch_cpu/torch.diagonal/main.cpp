#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }
    
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
        
        // Normalize dimensions to valid range for the tensor
        int64_t ndim = input_tensor.dim();
        if (ndim >= 2) {
            dim1 = dim1 % ndim;
            dim2 = dim2 % ndim;
            // Ensure dim1 != dim2
            if (dim1 == dim2) {
                dim2 = (dim1 + 1) % ndim;
            }
        }
        
        // Try different variants of the diagonal operation
        
        // Variant 1: Basic diagonal with default parameters
        try {
            torch::Tensor result1 = input_tensor.diagonal();
            (void)result1;
        } catch (const std::exception&) {
            // Catch and continue to next variant
        }
        
        // Variant 2: Diagonal with offset parameter
        try {
            torch::Tensor result2 = input_tensor.diagonal(offset_param);
            (void)result2;
        } catch (const std::exception&) {
            // Catch and continue to next variant
        }
        
        // Variant 3: Diagonal with offset and dim1 parameters
        try {
            torch::Tensor result3 = input_tensor.diagonal(offset_param, dim1);
            (void)result3;
        } catch (const std::exception&) {
            // Catch and continue to next variant
        }
        
        // Variant 4: Diagonal with all parameters
        try {
            torch::Tensor result4 = input_tensor.diagonal(offset_param, dim1, dim2);
            (void)result4;
        } catch (const std::exception&) {
            // Catch and continue
        }
        
        // Variant 5: Using functional API
        try {
            torch::Tensor result5 = torch::diagonal(input_tensor, offset_param, dim1, dim2);
            (void)result5;
        } catch (const std::exception&) {
            // Catch and continue
        }
        
        // Variant 6: Using negative dimensions
        try {
            int64_t neg_dim1 = -1;
            int64_t neg_dim2 = -2;
            torch::Tensor result6 = input_tensor.diagonal(offset_param, neg_dim1, neg_dim2);
            (void)result6;
        } catch (const std::exception&) {
            // Catch and continue
        }
        
        // Variant 7: Using various offset values
        try {
            // Use bounded offset based on fuzzer data
            int64_t bounded_offset = offset_param % 100;
            torch::Tensor result7 = input_tensor.diagonal(bounded_offset);
            (void)result7;
        } catch (const std::exception&) {
            // Catch and continue
        }
        
        // Variant 8: Negative offset
        try {
            int64_t neg_offset = -(std::abs(offset_param) % 100);
            torch::Tensor result8 = input_tensor.diagonal(neg_offset);
            (void)result8;
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