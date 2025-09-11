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
        
        // Need at least some data to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create a quantized per-channel tensor
        torch::Tensor quantized_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Try to get the zero points from the quantized tensor
        try {
            // q_per_channel_zero_points expects a quantized tensor with per-channel quantization
            // We'll try to call it regardless of the tensor type to test error handling
            torch::Tensor zero_points = torch::q_per_channel_zero_points(quantized_tensor);
            
            // If we get here, check that the zero points tensor has the expected properties
            if (zero_points.defined()) {
                // Access some properties to ensure the tensor is valid
                auto dtype = zero_points.dtype();
                auto numel = zero_points.numel();
                auto sizes = zero_points.sizes();
                
                // Try to access the data
                if (numel > 0) {
                    zero_points.item<int64_t>();
                }
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected and should not terminate the fuzzer
            return 0;
        }
        
        // If there's more data, try with different axis parameter
        if (offset + 1 < Size) {
            int64_t axis = static_cast<int64_t>(Data[offset++]) % std::max<int64_t>(1, quantized_tensor.dim());
            
            try {
                // Create a new tensor with the remaining data
                if (offset < Size) {
                    torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Try to get zero points with the axis parameter
                    torch::Tensor zero_points = torch::q_per_channel_zero_points(another_tensor);
                    
                    // Access properties to ensure the tensor is valid
                    if (zero_points.defined()) {
                        auto dtype = zero_points.dtype();
                        auto numel = zero_points.numel();
                        auto sizes = zero_points.sizes();
                    }
                }
            } catch (const c10::Error& e) {
                // PyTorch-specific exceptions are expected
                return 0;
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
