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
        
        // Need at least 1 byte for dtype selection
        if (Size < 1) {
            return 0;
        }
        
        // Parse the dtype to set for autocast
        uint8_t dtype_selector = Data[offset++];
        torch::ScalarType autocast_dtype = fuzzer_utils::parseDataType(dtype_selector);
        
        // Set the autocast GPU dtype
        torch::set_autocast_gpu_dtype(autocast_dtype);
        
        // Create a tensor to test with autocast
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test autocast functionality by creating a context and performing an operation
            bool enabled = offset < Size ? (Data[offset++] % 2 == 0) : true;
            
            // Create an autocast context
            {
                torch::AutocastMode guard(torch::kCUDA, enabled);
                
                // Perform some operations that would be affected by autocast
                if (tensor.defined()) {
                    // Simple operation that would be affected by autocast
                    torch::Tensor result = tensor + tensor;
                    
                    // Check if the result has the expected dtype when autocast is enabled
                    if (enabled && tensor.is_cuda() && 
                        (tensor.scalar_type() == torch::kFloat || 
                         tensor.scalar_type() == torch::kDouble)) {
                        // The result should have the autocast dtype
                        if (result.scalar_type() != autocast_dtype) {
                            // This is not an error, just a verification
                        }
                    }
                }
            }
            
            // Test nested autocast contexts with different settings
            if (offset < Size) {
                bool nested_enabled = Data[offset++] % 2 == 0;
                
                {
                    torch::AutocastMode outer_guard(torch::kCUDA, enabled);
                    
                    // Perform operation in outer context
                    torch::Tensor outer_result = tensor + tensor;
                    
                    {
                        // Create nested context with different setting
                        torch::AutocastMode inner_guard(torch::kCUDA, nested_enabled);
                        
                        // Perform operation in inner context
                        torch::Tensor inner_result = tensor + tensor;
                    }
                    
                    // Back to outer context
                    torch::Tensor after_nested = tensor + tensor;
                }
            }
            
            // Test changing the dtype during execution
            if (offset < Size) {
                uint8_t new_dtype_selector = Data[offset++];
                torch::ScalarType new_autocast_dtype = fuzzer_utils::parseDataType(new_dtype_selector);
                
                // Change the autocast dtype
                torch::set_autocast_gpu_dtype(new_autocast_dtype);
                
                // Test with the new dtype
                {
                    torch::AutocastMode guard(torch::kCUDA, true);
                    torch::Tensor result = tensor + tensor;
                }
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
