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
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply Hardswish using functional API
        torch::Tensor output = torch::hardswish(input);
        
        // Try inplace version if there's enough data to determine whether to use it
        if (offset < Size) {
            bool use_inplace = Data[offset++] % 2 == 0;
            if (use_inplace) {
                torch::Tensor input_clone = input.clone();
                torch::hardswish_(input_clone);
            }
        }
        
        // Try with different tensor options if there's more data
        if (offset + 1 < Size) {
            // Create a new tensor with different options
            size_t new_offset = offset;
            torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, new_offset);
            
            // Apply Hardswish to this new tensor
            torch::Tensor another_output = torch::hardswish(another_input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
