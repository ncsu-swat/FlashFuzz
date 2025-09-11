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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create weight tensor
        torch::Tensor weight = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Call fbgemm_linear_quantize_weight with only the input tensor
        auto result = torch::fbgemm_linear_quantize_weight(weight);
        
        // Access the result to ensure it's computed
        auto quantized_weight = std::get<0>(result);
        auto scale = std::get<1>(result);
        auto zero_point = std::get<2>(result);
        auto qparams = std::get<3>(result);
        
        // Force evaluation to catch any errors
        quantized_weight.sum().item<float>();
        scale.sum().item<float>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
