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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for log_ndtr
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply the torch.special.log_ndtr operation
        torch::Tensor result = torch::special::log_ndtr(input);
        
        // Try to access the result to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            auto accessor = result.accessor<float, 1>();
            volatile float first_element = accessor[0];
            (void)first_element; // Prevent unused variable warning
        }
        
        // Try with out variant if we have enough data left
        if (offset + 2 < Size) {
            torch::Tensor output = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure output has same dtype and device as input
            output = output.to(input.dtype()).to(input.device());
            
            // Resize output to match input shape if needed
            if (output.sizes() != input.sizes()) {
                output.resize_as_(input);
            }
            
            // Call the out variant
            torch::special::log_ndtr_out(output, input);
        }
        
        // Try with different input types if we have enough data left
        if (offset + 2 < Size) {
            // Create another tensor with potentially different properties
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Apply log_ndtr to the second tensor
            torch::Tensor result2 = torch::special::log_ndtr(input2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
