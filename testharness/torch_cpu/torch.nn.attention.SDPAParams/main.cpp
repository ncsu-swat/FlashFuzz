#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create query, key, value tensors
        torch::Tensor query = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for key and value
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor key = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor value = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create optional attn_mask tensor
        torch::Tensor attn_mask;
        bool use_mask = false;
        
        if (offset < Size && Data[offset++] % 2 == 0) {
            use_mask = true;
            if (offset < Size) {
                attn_mask = fuzzer_utils::createTensor(Data, Size, offset);
            }
        }
        
        // Parse dropout probability
        double dropout_p = 0.0;
        if (offset + sizeof(float) <= Size) {
            float temp_dropout;
            std::memcpy(&temp_dropout, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Ensure dropout_p is between 0 and 1
            temp_dropout = std::abs(temp_dropout);
            dropout_p = temp_dropout - std::floor(temp_dropout);
        }
        
        // Parse is_causal flag
        bool is_causal = false;
        if (offset < Size) {
            is_causal = (Data[offset++] % 2 == 0);
        }
        
        // Parse scale factor
        std::optional<double> scale = std::nullopt;
        if (offset + sizeof(float) <= Size) {
            float temp_scale;
            std::memcpy(&temp_scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            scale = static_cast<double>(temp_scale);
        }
        
        // Apply the scaled_dot_product_attention operation
        torch::Tensor output;
        
        if (use_mask) {
            output = torch::scaled_dot_product_attention(
                query, key, value, attn_mask, dropout_p, is_causal, scale);
        } else {
            output = torch::scaled_dot_product_attention(
                query, key, value, {}, dropout_p, is_causal, scale);
        }
        
        // Perform some operation on the output to ensure it's used
        auto sum = output.sum();
        
        // Prevent compiler from optimizing away the computation
        if (sum.item<float>() == -12345.6789f) {
            std::cerr << "Unlikely sum value encountered" << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
