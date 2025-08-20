#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get min value from remaining data
        double min_value = 0.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&min_value, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Apply clamp_min operation
        torch::Tensor result = torch::clamp_min(input_tensor, min_value);
        
        // Try in-place version if there's more data
        if (offset < Size && Data[offset] % 2 == 0) {
            torch::Tensor input_copy = input_tensor.clone();
            input_copy.clamp_min_(min_value);
        }
        
        // Try with tensor min value if there's more data
        if (offset + 1 < Size) {
            // Create a second tensor to use as min value
            torch::Tensor min_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Only proceed if shapes are compatible or can be broadcast
            try {
                torch::Tensor result2 = torch::clamp_min(input_tensor, min_tensor);
                
                // Try in-place version
                if (offset < Size && Data[offset] % 2 == 0) {
                    torch::Tensor input_copy = input_tensor.clone();
                    input_copy.clamp_min_(min_tensor);
                }
            }
            catch (const std::exception&) {
                // Silently ignore broadcasting errors
            }
        }
        
        // Try with named tensor variant if there's more data
        if (offset < Size) {
            torch::Tensor result3 = torch::clamp_min(input_tensor, min_value);
        }
        
        // Try with different output dtypes if there's more data
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            try {
                torch::Tensor result4 = torch::clamp_min(input_tensor.to(dtype), min_value);
            }
            catch (const std::exception&) {
                // Silently ignore dtype compatibility errors
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