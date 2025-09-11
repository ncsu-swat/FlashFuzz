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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create query tensor
        torch::Tensor query = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create key tensor
        if (offset < Size) {
            torch::Tensor key = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Create value tensor
            if (offset < Size) {
                torch::Tensor value = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Parse attention mask if we have more data
                torch::Tensor attn_mask;
                bool has_attn_mask = false;
                if (offset < Size) {
                    attn_mask = fuzzer_utils::createTensor(Data, Size, offset);
                    has_attn_mask = true;
                }
                
                // Parse dropout probability
                double dropout_p = 0.0;
                if (offset + sizeof(float) <= Size) {
                    float temp_dropout;
                    std::memcpy(&temp_dropout, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    // Clamp dropout to valid range [0, 1]
                    dropout_p = static_cast<double>(std::max(0.0f, std::min(1.0f, temp_dropout)));
                }
                
                // Parse is_causal flag
                bool is_causal = false;
                if (offset < Size) {
                    is_causal = static_cast<bool>(Data[offset++] & 0x01);
                }
                
                // Parse scale factor
                std::optional<double> scale = std::nullopt;
                if (offset + sizeof(float) <= Size) {
                    float temp_scale;
                    std::memcpy(&temp_scale, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    scale = static_cast<double>(temp_scale);
                }
                
                // Call the attention function with different combinations of parameters
                try {
                    // Basic call with required parameters
                    auto result1 = torch::scaled_dot_product_attention(
                        query, key, value);
                    
                    // Call with attention mask
                    if (has_attn_mask) {
                        auto result2 = torch::scaled_dot_product_attention(
                            query, key, value, attn_mask);
                    }
                    
                    // Call with all parameters
                    std::optional<torch::Tensor> attn_mask_opt = has_attn_mask ? std::optional<torch::Tensor>(attn_mask) : std::nullopt;
                    auto result3 = torch::scaled_dot_product_attention(
                        query, key, value, attn_mask_opt, dropout_p, is_causal, scale);
                    
                } catch (const c10::Error& e) {
                    // PyTorch specific errors are expected and handled
                }
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
