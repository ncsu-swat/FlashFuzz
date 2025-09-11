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
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.to operation with mtia device
        // MTIA is a device type for mobile tensor inference acceleration
        try {
            torch::Device mtia_device(torch::kMTIA);
            torch::Tensor result = input_tensor.to(mtia_device);
            
            // Verify that the result has the same data as the input
            if (result.defined() && input_tensor.defined()) {
                // Check if shapes and dtypes match
                if (result.sizes() != input_tensor.sizes() || 
                    result.dtype() != input_tensor.dtype()) {
                    throw std::runtime_error("MTIA result tensor has different shape or dtype");
                }
                
                // For numeric tensors, check if values are preserved
                if (input_tensor.is_floating_point() || input_tensor.is_complex() || 
                    input_tensor.is_signed()) {
                    torch::Tensor cpu_result = result.to(torch::kCPU);
                    if (!torch::allclose(cpu_result, input_tensor)) {
                        throw std::runtime_error("MTIA result tensor has different values");
                    }
                }
            }
            
            // Try to use the result tensor to ensure it's valid
            if (result.defined()) {
                auto sizes = result.sizes();
                auto numel = result.numel();
                auto dtype = result.dtype();
                
                // Try some basic operations on the result tensor
                if (numel > 0) {
                    torch::Tensor sum = result.sum();
                    torch::Tensor mean = result.mean();
                }
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected for some inputs
            // We catch them to avoid crashing the fuzzer
        }
        
        // If there's more data, try with options
        if (offset + 1 < Size) {
            bool non_blocking = Data[offset++] & 0x1;
            
            try {
                torch::Device mtia_device(torch::kMTIA);
                torch::Tensor result = input_tensor.to(mtia_device, non_blocking);
                
                // Perform similar checks as above
                if (result.defined() && input_tensor.defined()) {
                    if (result.sizes() != input_tensor.sizes() || 
                        result.dtype() != input_tensor.dtype()) {
                        throw std::runtime_error("MTIA result tensor with options has different shape or dtype");
                    }
                }
            } catch (const c10::Error& e) {
                // Expected for some inputs
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
