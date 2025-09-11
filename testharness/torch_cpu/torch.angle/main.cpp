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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.angle operation
        torch::Tensor result = torch::angle(input_tensor);
        
        // Try different variants of the operation
        if (offset + 1 < Size) {
            // Use the next byte to decide if we should try a different variant
            uint8_t variant_selector = Data[offset++];
            
            // For complex tensors, try out_variant
            if (variant_selector % 2 == 0 && 
                (input_tensor.scalar_type() == torch::kComplexFloat || 
                 input_tensor.scalar_type() == torch::kComplexDouble)) {
                
                // Create output tensor with appropriate dtype for angle result (real-valued)
                auto out_options = torch::TensorOptions().dtype(
                    input_tensor.scalar_type() == torch::kComplexFloat ? 
                    torch::kFloat : torch::kDouble);
                
                torch::Tensor out_tensor = torch::empty_like(input_tensor, out_options);
                
                // Call the out variant
                torch::angle_out(out_tensor, input_tensor);
            }
        }
        
        // Try calling angle on non-complex tensors too (should work for real tensors)
        if (offset + 1 < Size) {
            uint8_t try_real = Data[offset++];
            if (try_real % 3 == 0) {
                // Create a real tensor
                torch::Tensor real_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                torch::Tensor real_result = torch::angle(real_tensor);
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
