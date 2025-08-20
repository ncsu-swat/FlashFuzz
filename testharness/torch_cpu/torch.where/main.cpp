#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 3) {
            return 0;
        }
        
        // Create condition tensor
        torch::Tensor condition = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for x tensor
        if (offset >= Size) {
            // Test with default tensors if not enough data
            torch::Tensor x = torch::ones_like(condition);
            torch::Tensor y = torch::zeros_like(condition);
            torch::Tensor result = torch::where(condition, x, y);
            return 0;
        }
        
        // Create x tensor
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for y tensor
        if (offset >= Size) {
            // Test with default tensor for y
            torch::Tensor y = torch::zeros_like(x);
            torch::Tensor result = torch::where(condition, x, y);
            return 0;
        }
        
        // Create y tensor
        torch::Tensor y = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.where operation
        torch::Tensor result = torch::where(condition, x, y);
        
        // Test scalar condition variant if possible
        if (offset + 1 < Size) {
            bool scalar_condition = Data[offset++] & 0x1;
            torch::Tensor scalar_condition_tensor = torch::tensor(scalar_condition);
            torch::Tensor scalar_result = torch::where(scalar_condition_tensor, x, y);
        }
        
        // Test element-wise condition with scalar x and y if possible
        if (offset + 2 < Size) {
            // Create scalar values for x and y
            float scalar_x = static_cast<float>(Data[offset++]);
            float scalar_y = static_cast<float>(Data[offset++]);
            
            // Test with scalar x and tensor y
            torch::Tensor result2 = torch::where(condition, scalar_x, y);
            
            // Test with tensor x and scalar y
            torch::Tensor result3 = torch::where(condition, x, scalar_y);
            
            // Test with scalar x and scalar y
            torch::Tensor result4 = torch::where(condition, scalar_x, scalar_y);
        }
        
        // Test with different condition shapes
        if (offset < Size) {
            // Try to create a condition with different shape
            torch::Tensor alt_condition;
            try {
                alt_condition = fuzzer_utils::createTensor(Data, Size, offset);
                torch::Tensor result5 = torch::where(alt_condition, x, y);
            } catch (const std::exception&) {
                // Ignore exceptions from shape mismatch
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