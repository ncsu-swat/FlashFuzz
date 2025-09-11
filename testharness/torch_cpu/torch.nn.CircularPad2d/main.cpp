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
        
        // Need at least a few bytes for basic operations
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 4 bytes left for padding values
        if (Size - offset < 4) {
            return 0;
        }
        
        // Extract padding values from the input data
        int64_t left = static_cast<int64_t>(Data[offset++]);
        int64_t right = static_cast<int64_t>(Data[offset++]);
        int64_t top = static_cast<int64_t>(Data[offset++]);
        int64_t bottom = static_cast<int64_t>(Data[offset++]);
        
        // Apply circular padding using torch::nn::functional::pad
        torch::Tensor output = torch::nn::functional::pad(input_tensor, 
            torch::nn::functional::PadFuncOptions({left, right, top, bottom}).mode(torch::kCircular));
        
        // Force evaluation of the output tensor
        auto output_size = output.sizes();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
