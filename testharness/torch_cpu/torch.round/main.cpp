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
        
        // Apply torch.round operation
        torch::Tensor rounded_tensor = torch::round(input_tensor);
        
        // Try different rounding modes if we have more data
        if (offset + 1 < Size) {
            uint8_t rounding_mode_selector = Data[offset++];
            
            // Select rounding mode based on the byte
            if (rounding_mode_selector % 3 == 0) {
                // Test with decimals parameter
                int64_t decimals = 0;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&decimals, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    
                    // Apply round with decimals
                    torch::Tensor rounded_with_decimals = torch::round(input_tensor, decimals);
                }
            } else if (rounding_mode_selector % 3 == 1) {
                // Test with out parameter
                torch::Tensor out_tensor = torch::empty_like(input_tensor);
                torch::round_out(out_tensor, input_tensor);
            }
        }
        
        // Try inplace version if we have more data
        if (offset < Size) {
            uint8_t inplace_selector = Data[offset++];
            
            if (inplace_selector % 2 == 0) {
                // Create a copy to avoid modifying the original tensor
                torch::Tensor inplace_tensor = input_tensor.clone();
                inplace_tensor.round_();
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
