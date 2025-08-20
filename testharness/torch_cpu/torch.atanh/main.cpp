#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for atanh
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply atanh operation
        torch::Tensor result = torch::atanh(input);
        
        // Try in-place version if there's more data
        if (offset < Size) {
            torch::Tensor input_copy = input.clone();
            input_copy.atanh_();
        }
        
        // Try with different options if we have more data
        if (offset + 1 < Size) {
            // Use the next byte to determine if we should try other variants
            uint8_t variant_selector = Data[offset++];
            
            // Try named parameter version
            if (variant_selector & 0x1) {
                torch::Tensor result_named = torch::atanh(input);
            }
            
            // Try out parameter version
            if (variant_selector & 0x2) {
                torch::Tensor out = torch::empty_like(input);
                torch::atanh_out(out, input);
            }
            
            // Try with different input types
            if (variant_selector & 0x4 && offset < Size) {
                try {
                    // Create another tensor with potentially different properties
                    torch::Tensor another_input = fuzzer_utils::createTensor(Data, Size, offset);
                    torch::Tensor another_result = torch::atanh(another_input);
                } catch (const std::exception &) {
                    // Ignore exceptions from the second tensor creation
                }
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