#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Try to move the tensor to XPU
        try {
            // Attempt to move tensor to XPU device
            torch::Tensor xpu_tensor = tensor.to(torch::Device(torch::kXPU));
            
            // Try to perform some operations on the XPU tensor
            if (xpu_tensor.defined()) {
                // Basic operations to test XPU functionality
                torch::Tensor result1 = xpu_tensor + 1;
                torch::Tensor result2 = xpu_tensor * 2;
                torch::Tensor result3 = torch::sin(xpu_tensor);
                
                // Move back to CPU for verification
                torch::Tensor cpu_result = result3.cpu();
            }
        } catch (const c10::Error& e) {
            // XPU might not be available, which is expected in many environments
            // Just catch and continue
        }
        
        // If we have more data, try creating another tensor and test more XPU operations
        if (offset + 2 < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            try {
                // Move both tensors to XPU and perform operations between them
                torch::Tensor xpu_tensor1 = tensor.to(torch::Device(torch::kXPU));
                torch::Tensor xpu_tensor2 = tensor2.to(torch::Device(torch::kXPU));
                
                // Try to perform operations if shapes are compatible
                if (xpu_tensor1.sizes() == xpu_tensor2.sizes()) {
                    torch::Tensor add_result = xpu_tensor1 + xpu_tensor2;
                    torch::Tensor mul_result = xpu_tensor1 * xpu_tensor2;
                    torch::Tensor div_result = xpu_tensor1 / (xpu_tensor2 + 0.1); // Avoid division by zero
                    
                    // Move results back to CPU
                    torch::Tensor cpu_result = div_result.cpu();
                }
                
                // Test other XPU operations
                if (xpu_tensor1.dim() > 0 && xpu_tensor1.size(0) > 0) {
                    torch::Tensor slice_result = xpu_tensor1.slice(0, 0, xpu_tensor1.size(0) / 2 + 1);
                    torch::Tensor cpu_slice = slice_result.cpu();
                }
                
                // Test type conversion on XPU
                torch::Tensor float_tensor = xpu_tensor1.to(torch::kFloat);
                torch::Tensor int_tensor = xpu_tensor1.to(torch::kInt);
                
                // Test reduction operations
                torch::Tensor sum_result = xpu_tensor1.sum();
                torch::Tensor mean_result = xpu_tensor1.mean();
                
                // Move results back to CPU
                torch::Tensor cpu_sum = sum_result.cpu();
                torch::Tensor cpu_mean = mean_result.cpu();
            } catch (const c10::Error& e) {
                // XPU operations might fail, which is expected in some cases
                // Just catch and continue
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