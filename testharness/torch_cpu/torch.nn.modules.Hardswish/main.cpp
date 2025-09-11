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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply Hardswish using functional API
        torch::Tensor output = torch::hardswish(input);
        
        // Try inplace version if available
        if (input.is_floating_point() && input.requires_grad() == false) {
            torch::Tensor input_copy = input.clone();
            torch::Tensor out_tensor = torch::empty_like(input_copy);
            torch::hardswish_out(out_tensor, input_copy);
        }
        
        // Try with different configurations
        if (offset + 1 < Size) {
            bool inplace = Data[offset++] % 2 == 0;
            
            // Apply the hardswish function again
            torch::Tensor output2 = torch::hardswish(input);
            
            // If inplace operation is requested and tensor allows it
            if (inplace && input.is_floating_point() && !input.requires_grad()) {
                torch::Tensor input_copy = input.clone();
                torch::hardswish(input_copy);
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
