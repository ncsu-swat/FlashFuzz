#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for the test
        if (Size < 1) {
            return 0;
        }
        
        // Parse a boolean value from the first byte to determine if we should enable autocast
        bool enable_autocast = Data[0] & 0x1;
        offset++;
        
        // Parse a value to determine the dtype to test
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            
            // Get different dtypes to test
            torch::ScalarType dtype;
            
            // Use the selector to choose between different dtypes
            switch (dtype_selector % 5) {
                case 0:
                    dtype = torch::kFloat;
                    break;
                case 1:
                    dtype = torch::kDouble;
                    break;
                case 2:
                    dtype = torch::kHalf;
                    break;
                case 3:
                    dtype = torch::kBFloat16;
                    break;
                case 4:
                default:
                    dtype = torch::kFloat32;
                    break;
            }
            
            // Test the get_autocast_gpu_dtype function
            if (enable_autocast) {
                torch::autocast::set_autocast_enabled(torch::kCUDA, true);
            } else {
                torch::autocast::set_autocast_enabled(torch::kCUDA, false);
            }
            
            // Get the autocast GPU dtype
            torch::ScalarType autocast_dtype = torch::get_autocast_gpu_dtype();
            
            // Create a tensor with the original dtype
            torch::Tensor tensor;
            if (offset < Size) {
                tensor = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                // Create a simple tensor if we don't have enough data
                tensor = torch::ones({2, 2}, torch::TensorOptions().dtype(dtype));
            }
            
            // Test casting the tensor to the autocast dtype
            if (tensor.defined()) {
                torch::Tensor casted_tensor = tensor.to(autocast_dtype);
            }
            
            // Reset autocast state
            torch::autocast::set_autocast_enabled(torch::kCUDA, false);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}