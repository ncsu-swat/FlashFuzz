#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

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
        
        // Parse the dtype to use for autocast
        uint8_t dtype_selector = Data[offset++];
        torch::ScalarType autocast_dtype = fuzzer_utils::parseDataType(dtype_selector);
        
        // Create a tensor to test with autocast
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Test autocast by performing an operation that might use the autocast dtype
            torch::Tensor result = tensor + tensor;
            
            // Verify the result exists (no crash)
            bool is_valid = result.defined();
            if (!is_valid) {
                return 0;
            }
            
            // Test with different device types if available
            if (torch::cuda::is_available() && offset < Size) {
                // Use one more byte to decide device type
                bool use_cuda = (offset < Size) ? (Data[offset++] % 2 == 0) : false;
                
                if (use_cuda) {
                    // Create a CUDA tensor and test operations
                    if (offset < Size) {
                        torch::Tensor cuda_tensor = fuzzer_utils::createTensor(Data, Size, offset).cuda();
                        torch::Tensor cuda_result = cuda_tensor + cuda_tensor;
                        bool cuda_valid = cuda_result.defined();
                        if (!cuda_valid) {
                            return 0;
                        }
                    }
                }
            }
            
            // Test enabling/disabling autocast using autocast context manager
            if (offset < Size) {
                bool enable_autocast = (Data[offset++] % 2 == 0);
                
                if (enable_autocast) {
                    // Use autocast context manager
                    torch::autocast autocast_guard(torch::kCUDA, true);
                    
                    // Perform operation with autocast enabled
                    torch::Tensor mode_result = tensor * 2.0;
                    bool mode_valid = mode_result.defined();
                    if (!mode_valid) {
                        return 0;
                    }
                } else {
                    // Perform operation without autocast
                    torch::Tensor mode_result = tensor * 2.0;
                    bool mode_valid = mode_result.defined();
                    if (!mode_valid) {
                        return 0;
                    }
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