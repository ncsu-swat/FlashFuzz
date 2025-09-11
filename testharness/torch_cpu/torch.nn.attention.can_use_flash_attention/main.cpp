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
        
        // If we have more data, create key and value tensors
        torch::Tensor key;
        torch::Tensor value;
        
        if (offset < Size) {
            key = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            key = query.clone();
        }
        
        if (offset < Size) {
            value = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            value = key.clone();
        }
        
        // Parse additional parameters if data available
        bool need_weights = true;
        float dropout_p = 0.0;
        bool is_causal = false;
        float scale = 1.0;
        
        if (offset + 4 <= Size) {
            need_weights = Data[offset++] % 2 == 0;
            
            // Parse dropout_p as a float between 0 and 1
            if (offset + sizeof(float) <= Size) {
                float raw_dropout;
                std::memcpy(&raw_dropout, Data + offset, sizeof(float));
                offset += sizeof(float);
                dropout_p = std::abs(raw_dropout) / (std::abs(raw_dropout) + 1.0f); // Normalize to [0,1]
            }
            
            is_causal = Data[offset++] % 2 == 0;
            
            // Parse scale as a float
            if (offset + sizeof(float) <= Size) {
                float raw_scale;
                std::memcpy(&raw_scale, Data + offset, sizeof(float));
                offset += sizeof(float);
                scale = raw_scale;
            }
        }
        
        // Call the can_use_flash_attention function
        bool can_use_flash = torch::can_use_flash_attention(
            query, key, value, need_weights, dropout_p, is_causal, scale);
        
        // Use the result to prevent optimization
        if (can_use_flash) {
            torch::Tensor dummy = torch::ones({1});
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
