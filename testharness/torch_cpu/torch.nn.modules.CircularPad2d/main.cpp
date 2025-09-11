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
        
        // Need at least 1 byte for padding configuration
        if (Size < 1) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor;
        if (offset < Size) {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            return 0;
        }
        
        // Extract padding values from the remaining data
        int64_t padding[4] = {1, 1, 1, 1}; // Default padding
        
        for (int i = 0; i < 4 && offset + sizeof(int64_t) <= Size; i++) {
            int64_t pad_value;
            std::memcpy(&pad_value, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            padding[i] = pad_value;
        }
        
        // Apply circular padding using functional interface
        torch::Tensor output = torch::nn::functional::pad(
            input_tensor, 
            {padding[0], padding[1], padding[2], padding[3]},
            torch::nn::functional::PadFuncOptions().mode(torch::kCircular)
        );
        
        // Ensure the output is materialized
        output.sum().item<float>();
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
