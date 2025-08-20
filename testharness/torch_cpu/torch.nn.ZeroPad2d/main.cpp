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
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 4 bytes left for padding values
        if (Size - offset < 4) {
            return 0;
        }
        
        // Extract padding values from the remaining data
        int64_t left = static_cast<int64_t>(Data[offset++]);
        int64_t right = static_cast<int64_t>(Data[offset++]);
        int64_t top = static_cast<int64_t>(Data[offset++]);
        int64_t bottom = static_cast<int64_t>(Data[offset++]);
        
        // Create padding vector
        std::vector<int64_t> padding = {left, right, top, bottom};
        
        // Create ZeroPad2d module
        torch::nn::ZeroPad2d zero_pad(padding);
        
        // Apply padding to the input tensor
        torch::Tensor output_tensor = zero_pad(input_tensor);
        
        // Alternative approach: use functional interface
        torch::Tensor output_tensor2 = torch::nn::functional::pad(
            input_tensor,
            torch::nn::functional::PadFuncOptions({left, right, top, bottom}).mode(torch::kConstant)
        );
        
        // Force computation to ensure any errors are triggered
        output_tensor.sum().item<float>();
        output_tensor2.sum().item<float>();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}