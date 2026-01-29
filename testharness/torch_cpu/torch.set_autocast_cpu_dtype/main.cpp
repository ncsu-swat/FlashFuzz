#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>
#include <ATen/autocast_mode.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for dtype selection
        if (Size < 1) {
            return 0;
        }
        
        // Parse the dtype to set for autocast
        uint8_t dtype_selector = Data[offset++];
        
        // Limit to dtypes that are valid for autocast CPU
        // Typically bfloat16 or float16 for autocast
        torch::ScalarType autocast_dtype;
        switch (dtype_selector % 3) {
            case 0:
                autocast_dtype = torch::kBFloat16;
                break;
            case 1:
                autocast_dtype = torch::kHalf;
                break;
            default:
                autocast_dtype = torch::kFloat;
                break;
        }
        
        // Set the autocast CPU dtype using the new API
        at::autocast::set_autocast_dtype(at::kCPU, autocast_dtype);
        
        // Create a tensor to test with autocast
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure tensor is float type for autocast operations
            if (!tensor.is_floating_point()) {
                tensor = tensor.to(torch::kFloat);
            }
            
            // Test autocast functionality
            {
                at::autocast::set_autocast_enabled(at::kCPU, true);
                
                // Perform some operations that would trigger autocast
                // Use inner try-catch for shape-related failures
                try {
                    torch::Tensor result = tensor + tensor;
                    torch::Tensor result2 = tensor * 2.0f;
                    
                    // For matmul, create compatible shapes
                    if (tensor.dim() == 2 && tensor.size(0) > 0 && tensor.size(1) > 0) {
                        torch::Tensor transposed = tensor.t();
                        torch::Tensor result3 = torch::matmul(tensor, transposed);
                    } else if (tensor.dim() == 1 && tensor.size(0) > 0) {
                        torch::Tensor result3 = torch::dot(tensor, tensor);
                    }
                } catch (...) {
                    // Silently catch shape mismatches
                }
                
                at::autocast::set_autocast_enabled(at::kCPU, false);
            }
            
            // Test with autocast disabled
            {
                at::autocast::set_autocast_enabled(at::kCPU, false);
                
                try {
                    torch::Tensor result = tensor + tensor;
                    torch::Tensor result2 = tensor * 2.0f;
                } catch (...) {
                    // Silently catch failures
                }
            }
            
            // Test toggling autocast multiple times
            {
                at::autocast::set_autocast_enabled(at::kCPU, true);
                
                try {
                    torch::Tensor outer_result = tensor + tensor;
                } catch (...) {}
                
                at::autocast::set_autocast_enabled(at::kCPU, false);
                
                try {
                    torch::Tensor inner_result = tensor - tensor;
                } catch (...) {}
                
                at::autocast::set_autocast_enabled(at::kCPU, true);
                
                try {
                    torch::Tensor outer_result2 = tensor * 2.0f;
                } catch (...) {}
                
                at::autocast::set_autocast_enabled(at::kCPU, false);
            }
        }
        
        // Get the current autocast dtype to verify it was set
        torch::ScalarType current_dtype = at::autocast::get_autocast_dtype(at::kCPU);
        (void)current_dtype; // Avoid unused variable warning
        
        // Reset autocast dtype to default
        at::autocast::set_autocast_dtype(at::kCPU, torch::kBFloat16);
        at::autocast::set_autocast_enabled(at::kCPU, false);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}