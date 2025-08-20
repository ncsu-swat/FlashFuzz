#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create first input tensor
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second input tensor if we have more data
        torch::Tensor input2;
        if (offset < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If not enough data for second tensor, create a tensor with same shape but different values
            input2 = torch::ones_like(input1);
        }
        
        // Apply torch.fmax operation
        torch::Tensor result = torch::fmax(input1, input2);
        
        // Try element-wise variant if we have more data
        if (offset < Size && Size - offset > 0) {
            // Create a scalar tensor from remaining data
            uint8_t scalar_byte = Data[offset++];
            double scalar_value = static_cast<double>(scalar_byte) / 255.0 * 100.0;
            torch::Tensor scalar_tensor = torch::scalar_tensor(scalar_value);
            
            // Apply torch.fmax with scalar tensor
            torch::Tensor scalar_result = torch::fmax(input1, scalar_tensor);
        }
        
        // Try other variants if we have more data
        if (offset < Size && Size - offset > 2) {
            // Create tensors with different shapes to test broadcasting
            std::vector<int64_t> new_shape;
            uint8_t rank_byte = Data[offset++];
            uint8_t rank = fuzzer_utils::parseRank(rank_byte);
            
            if (rank > 0) {
                new_shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
                torch::Tensor broadcast_tensor = torch::ones(new_shape);
                
                // Try broadcasting
                torch::Tensor broadcast_result = torch::fmax(input1, broadcast_tensor);
            }
        }
        
        // Try with empty tensors
        torch::Tensor empty_tensor = torch::empty({0});
        try {
            torch::Tensor empty_result = torch::fmax(empty_tensor, empty_tensor);
        } catch (...) {
            // Expected exception for empty tensors
        }
        
        // Try with NaN values
        torch::Tensor nan_tensor = torch::full_like(input1, std::numeric_limits<float>::quiet_NaN());
        torch::Tensor nan_result = torch::fmax(input1, nan_tensor);
        
        // Try with infinity values
        torch::Tensor inf_tensor = torch::full_like(input1, std::numeric_limits<float>::infinity());
        torch::Tensor inf_result = torch::fmax(input1, inf_tensor);
        
        // Try with negative infinity
        torch::Tensor neg_inf_tensor = torch::full_like(input1, -std::numeric_limits<float>::infinity());
        torch::Tensor neg_inf_result = torch::fmax(input1, neg_inf_tensor);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}