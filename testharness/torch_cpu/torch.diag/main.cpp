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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract a diagonal or construct a diagonal matrix
        if (offset + 1 < Size) {
            // Get a value for diagonal offset parameter
            int64_t diagonal = static_cast<int8_t>(Data[offset++]);
            
            // Test torch::diag with the input tensor
            torch::Tensor result = torch::diag(input_tensor, diagonal);
            
            // Test edge case: apply diag to the result again
            if (offset < Size && Data[offset++] % 2 == 0) {
                int64_t second_diagonal = static_cast<int8_t>(Data[offset++] % 10);
                torch::Tensor second_result = torch::diag(result, second_diagonal);
            }
        } else {
            // If we don't have enough data for the diagonal parameter,
            // just use the default diagonal (0)
            torch::Tensor result = torch::diag(input_tensor);
        }
        
        // Test with different diagonal values if we have more data
        if (offset + 2 < Size) {
            int64_t large_diagonal = *reinterpret_cast<const int32_t*>(Data + offset);
            offset += sizeof(int32_t);
            
            // Try with a potentially large diagonal value
            torch::Tensor result_large_diag = torch::diag(input_tensor, large_diagonal);
        }
        
        // Test with negative diagonal values
        if (offset < Size) {
            int64_t negative_diagonal = -static_cast<int8_t>(Data[offset++]);
            
            // Try with a negative diagonal value
            torch::Tensor result_neg_diag = torch::diag(input_tensor, negative_diagonal);
        }
        
        // Test with empty tensor if we have more data
        if (offset < Size && Data[offset++] % 3 == 0) {
            std::vector<int64_t> empty_shape = {0};
            torch::Tensor empty_tensor = torch::empty(empty_shape);
            torch::Tensor empty_result = torch::diag(empty_tensor);
        }
        
        // Test with 0-dimensional tensor if we have more data
        if (offset < Size && Data[offset++] % 3 == 1) {
            torch::Tensor scalar_tensor = torch::tensor(42);
            torch::Tensor scalar_result = torch::diag(scalar_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
