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
        
        // Skip if not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.values_copy operation
        torch::Tensor result = torch::values_copy(input_tensor);
        
        // Try to access the values to ensure computation is performed
        if (result.defined() && result.numel() > 0) {
            auto item = result.item();
        }
        
        // Try with different options
        if (offset + 1 < Size) {
            // Create a second tensor if we have more data
            torch::Tensor second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test values_copy with second tensor (separate call)
            torch::Tensor second_result = torch::values_copy(second_tensor);
            
            // Try with non-contiguous tensors
            if (input_tensor.dim() > 1 && input_tensor.size(0) > 1) {
                torch::Tensor non_contiguous = input_tensor.transpose(0, input_tensor.dim() - 1);
                torch::values_copy(non_contiguous);
            }
            
            // Try with different memory formats
            if (input_tensor.dim() == 4) {
                torch::Tensor channels_last = input_tensor.to(torch::MemoryFormat::ChannelsLast);
                torch::values_copy(channels_last);
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
