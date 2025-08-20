#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/csrc/autograd/autocast_mode.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for dtype selection
        if (Size < 1) {
            return 0;
        }
        
        // Parse the dtype to set for autocast
        uint8_t dtype_selector = Data[offset++];
        torch::ScalarType autocast_dtype = fuzzer_utils::parseDataType(dtype_selector);
        
        // Set the autocast CPU dtype
        torch::autograd::set_autocast_cpu_dtype(autocast_dtype);
        
        // Create a tensor to test with autocast
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test autocast functionality by creating a context and performing an operation
            {
                torch::autograd::AutocastMode autocast_mode(torch::kCPU, true);
                
                // Perform some operations that would trigger autocast
                torch::Tensor result = tensor + tensor;
                torch::Tensor result2 = tensor * 2.0;
                if (tensor.dim() >= 2) {
                    torch::Tensor result3 = torch::matmul(tensor, tensor);
                }
            }
            
            // Test disabling autocast
            {
                torch::autograd::AutocastMode autocast_mode(torch::kCPU, false);
                
                // Perform operations with autocast disabled
                torch::Tensor result = tensor + tensor;
                torch::Tensor result2 = tensor * 2.0;
            }
            
            // Test nested autocast contexts
            {
                torch::autograd::AutocastMode outer_mode(torch::kCPU, true);
                
                // Perform operation in outer context
                torch::Tensor outer_result = tensor + tensor;
                
                {
                    // Inner context with different setting
                    torch::autograd::AutocastMode inner_mode(torch::kCPU, false);
                    torch::Tensor inner_result = tensor + tensor;
                }
                
                // Back to outer context
                torch::Tensor outer_result2 = tensor * 2.0;
            }
        }
        
        // Reset autocast dtype to default (float32)
        torch::autograd::set_autocast_cpu_dtype(torch::ScalarType::Float);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}