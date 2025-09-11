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
        
        if (Size < 4) {
            return 0;
        }
        
        // Create query, key, value tensors
        torch::Tensor query = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor key = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        torch::Tensor value = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse attn_mask if there's data left
        torch::Tensor attn_mask;
        if (offset < Size) {
            attn_mask = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Parse dropout_p if there's data left
        double dropout_p = 0.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&dropout_p, Data + offset, sizeof(double));
            offset += sizeof(double);
            dropout_p = std::abs(dropout_p) / 10.0; // Normalize to [0, 0.1]
        }
        
        // Parse is_causal if there's data left
        bool is_causal = false;
        if (offset < Size) {
            is_causal = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Parse scale if there's data left
        c10::optional<double> scale = c10::nullopt;
        if (offset + sizeof(double) <= Size) {
            double scale_val;
            std::memcpy(&scale_val, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Avoid extreme values that might cause numerical issues
            scale_val = std::max(0.01, std::min(100.0, std::abs(scale_val)));
            scale = scale_val;
        }
        
        // Try to apply scaled_dot_product_attention
        try {
            torch::Tensor output = torch::scaled_dot_product_attention(
                query, key, value, attn_mask, dropout_p, is_causal, scale);
        } catch (const c10::Error& e) {
            // Expected exceptions from PyTorch are fine
        }
        
        // Try with different parameters
        try {
            torch::Tensor output = torch::scaled_dot_product_attention(
                query, key, value, {}, 0.0, false, c10::nullopt);
        } catch (const c10::Error& e) {
            // Expected exceptions from PyTorch are fine
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
