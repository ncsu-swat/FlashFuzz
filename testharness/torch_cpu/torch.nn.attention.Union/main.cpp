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
            // Test with just query tensor
            auto result = torch::nn::functional::scaled_dot_product_attention(
                query, query, query);
            return 0;
        }
        
        torch::Tensor key = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            // Test with query and key tensors
            auto result = torch::nn::functional::scaled_dot_product_attention(
                query, key, key);
            return 0;
        }
        
        torch::Tensor value = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create optional parameters if we have more data
        torch::optional<torch::Tensor> attn_mask = torch::nullopt;
        if (offset + 2 < Size) {
            attn_mask = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Create dropout probability
        double dropout_p = 0.0;
        if (offset + sizeof(float) <= Size) {
            float dropout_val;
            std::memcpy(&dropout_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure dropout is between 0 and 1
            dropout_p = std::abs(dropout_val) / (std::abs(dropout_val) + 1.0);
        }
        
        // Create is_causal flag
        bool is_causal = false;
        if (offset < Size) {
            is_causal = Data[offset++] & 0x1;
        }
        
        // Create scale factor
        torch::optional<double> scale = torch::nullopt;
        if (offset + sizeof(float) <= Size) {
            float scale_val;
            std::memcpy(&scale_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            scale = static_cast<double>(scale_val);
        }
        
        // Apply the attention operation
        auto result = torch::nn::functional::scaled_dot_product_attention(
            query, key, value, attn_mask, dropout_p, is_causal, scale);
        
        // Test tensor operations instead of non-existent Union
        if (offset < Size) {
            // Create a second attention result to test operations with
            auto result2 = torch::nn::functional::scaled_dot_product_attention(
                query, key, value);
            
            // Test tensor concatenation as an alternative to Union
            auto union_result = torch::cat({result, result2}, 0);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
