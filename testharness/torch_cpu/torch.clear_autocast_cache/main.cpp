#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Create a tensor to ensure PyTorch is initialized
        if (Size > 0) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Test autocast functionality
        bool enabled = false;
        if (offset < Size) {
            enabled = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Set up autocast context
        if (enabled) {
            torch::autocast::set_enabled(torch::kCPU, true);
            
            // Create some tensors with autocast enabled
            if (offset < Size) {
                torch::Tensor t1 = fuzzer_utils::createTensor(Data, Size, offset);
                
                if (offset < Size) {
                    torch::Tensor t2 = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Perform some operations that might use the autocast cache
                    torch::Tensor result = t1 + t2;
                    result = torch::matmul(result, t1);
                }
            }
        }
        
        // Test the clear_autocast_cache function
        torch::clear_autocast_cache();
        
        // Try again with different device type if available
        if (offset < Size && torch::cuda::is_available()) {
            bool use_cuda = static_cast<bool>(Data[offset++] & 0x01);
            
            if (use_cuda) {
                torch::autocast::set_enabled(torch::kCUDA, true);
                
                // Create CUDA tensors and perform operations
                if (offset < Size) {
                    torch::Tensor t1 = fuzzer_utils::createTensor(Data, Size, offset).to(torch::kCUDA);
                    
                    if (offset < Size) {
                        torch::Tensor t2 = fuzzer_utils::createTensor(Data, Size, offset).to(torch::kCUDA);
                        
                        // Perform operations that might use the autocast cache
                        torch::Tensor result = t1 * t2;
                        result = torch::nn::functional::relu(result);
                    }
                }
                
                // Clear the CUDA autocast cache
                torch::clear_autocast_cache();
            }
        }
        
        // Test with different dtype
        if (offset < Size) {
            torch::autocast::set_enabled(torch::kCPU, true);
            
            uint8_t dtype_selector = Data[offset++];
            torch::ScalarType dtype;
            
            if (dtype_selector % 2 == 0) {
                dtype = torch::kFloat16;
            } else {
                dtype = torch::kBFloat16;
            }
            
            torch::autocast::set_autocast_dtype(torch::kCPU, dtype);
            
            // Create tensors and perform operations with the new dtype
            if (offset < Size) {
                torch::Tensor t1 = fuzzer_utils::createTensor(Data, Size, offset);
                
                if (offset < Size) {
                    torch::Tensor t2 = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Perform operations that might use the autocast cache
                    torch::Tensor result = torch::add(t1, t2);
                    result = torch::sin(result);
                }
            }
            
            // Clear the cache again
            torch::clear_autocast_cache();
        }
        
        // Reset autocast state
        torch::autocast::set_enabled(torch::kCPU, false);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}