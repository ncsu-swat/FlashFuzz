#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensors for testing contextlib
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a second tensor if there's enough data left
        torch::Tensor tensor2;
        if (offset + 2 < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            tensor2 = torch::ones_like(tensor1);
        }
        
        // Test basic tensor operations since torch.nn.attention.contextlib
        // doesn't have a direct C++ equivalent
        
        // Perform some attention-related operations
        auto result1 = torch::matmul(tensor1, tensor2);
        
        // Test with scaled_dot_product_attention if tensors have compatible shapes
        try {
            if (tensor1.dim() >= 2 && tensor2.dim() >= 2) {
                // Reshape tensors to be compatible with attention operations if needed
                auto q = tensor1.reshape({1, 1, -1, tensor1.size(-1)});
                auto k = tensor2.reshape({1, 1, -1, tensor2.size(-1)});
                auto v = tensor1.reshape({1, 1, -1, tensor1.size(-1)});
                
                // Try scaled dot product attention
                auto attn_output = torch::scaled_dot_product_attention(
                    q, k, v, 
                    torch::Tensor(), // attn_mask
                    0.0,             // dropout_p
                    false            // is_causal
                );
            }
        } catch (const std::exception&) {
            // Ignore exceptions from the attention operation itself
        }
        
        // Test with another operation
        auto result2 = torch::add(tensor1, tensor2);
        
        // Try nested operations to simulate context behavior
        {
            auto result3 = torch::mul(tensor1, tensor2);
        }
        
        // Final operation
        auto final_result = torch::sub(tensor1, tensor2);
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}