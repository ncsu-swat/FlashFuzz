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
        
        // Create query, key, value tensors
        torch::Tensor query = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for key and value
        if (offset >= Size) {
            torch::nn::functional::scaled_dot_product_attention(query, query, query);
            return 0;
        }
        
        torch::Tensor key = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            torch::nn::functional::scaled_dot_product_attention(query, key, key);
            return 0;
        }
        
        torch::Tensor value = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create optional parameters based on remaining data
        bool is_causal = false;
        float dropout_p = 0.0;
        
        if (offset < Size) {
            is_causal = Data[offset++] % 2 == 0;
        }
        
        if (offset < Size) {
            // Use the next byte to determine dropout probability (0.0 to 1.0)
            dropout_p = static_cast<float>(Data[offset++]) / 255.0f;
        }
        
        // Call the scaled_dot_product_attention function
        auto options = torch::nn::functional::ScaledDotProductAttentionFuncOptions()
            .dropout_p(dropout_p)
            .is_causal(is_causal);
            
        torch::nn::functional::scaled_dot_product_attention(query, key, value, torch::nullopt, options);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
