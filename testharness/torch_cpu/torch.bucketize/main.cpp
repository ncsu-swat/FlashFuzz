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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create boundaries tensor
        torch::Tensor boundaries = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for bucketize
        bool out_int32 = false;
        bool right = false;
        
        // Use remaining bytes to determine parameters if available
        if (offset + 1 < Size) {
            out_int32 = Data[offset++] & 0x1;
        }
        
        if (offset < Size) {
            right = Data[offset++] & 0x1;
        }
        
        // Apply bucketize operation
        torch::Tensor result = torch::bucketize(input, boundaries, out_int32, right);
        
        // Try different variants of the API
        if (offset < Size) {
            // Try the out variant if we have more data
            torch::Tensor output = torch::empty_like(input, 
                out_int32 ? torch::kInt32 : torch::kInt64);
            torch::bucketize_out(output, input, boundaries, out_int32, right);
        }
        
        // Try with different parameters if we have more data
        if (offset < Size) {
            bool new_right = Data[offset++] & 0x1;
            torch::Tensor result2 = torch::bucketize(input, boundaries, out_int32, new_right);
        }
        
        // Try with different out_int32 parameter
        if (offset < Size) {
            bool new_out_int32 = Data[offset++] & 0x1;
            torch::Tensor result3 = torch::bucketize(input, boundaries, new_out_int32, right);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
