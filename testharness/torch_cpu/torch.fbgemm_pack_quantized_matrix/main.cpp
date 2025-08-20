#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get parameters for fbgemm_pack_quantized_matrix
        // We need K, N for the matrix dimensions
        int64_t K = 0;
        int64_t N = 0;
        
        if (input_tensor.dim() >= 2) {
            K = input_tensor.size(0);
            N = input_tensor.size(1);
        } else if (input_tensor.dim() == 1) {
            K = 1;
            N = input_tensor.size(0);
        } else {
            // For scalar tensors, use small dimensions
            K = 1;
            N = 1;
        }
        
        // Convert input tensor to uint8 if needed
        torch::Tensor uint8_input;
        if (input_tensor.scalar_type() != torch::kUInt8) {
            uint8_input = input_tensor.to(torch::kUInt8);
        } else {
            uint8_input = input_tensor;
        }
        
        // Call fbgemm_pack_quantized_matrix with correct signature
        torch::Tensor packed_weights;
        try {
            packed_weights = torch::fbgemm_pack_quantized_matrix(
                uint8_input,
                K,
                N
            );
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors but don't discard the input
            return 0;
        }
        
        // Try to use the packed weights to ensure they're valid
        if (packed_weights.defined()) {
            auto sizes = packed_weights.sizes();
            auto numel = packed_weights.numel();
            auto dtype = packed_weights.dtype();
            
            // Access some elements to ensure the tensor is valid
            if (numel > 0) {
                auto first_elem = packed_weights.flatten()[0].item<float>();
                auto last_elem = packed_weights.flatten()[numel - 1].item<float>();
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