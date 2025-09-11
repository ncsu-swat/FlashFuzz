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
        
        // Parse the dtype to use for autocast
        uint8_t dtype_selector = Data[offset++];
        torch::ScalarType autocast_dtype = fuzzer_utils::parseDataType(dtype_selector);
        
        // Set the autocast IPU dtype
        torch::autocast::set_autocast_ipu_dtype(autocast_dtype);
        
        // Create a tensor to test with the autocast setting
        if (offset < Size) {
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Perform some operation that might be affected by autocast
            torch::Tensor result = tensor + tensor;
            
            // Reset autocast dtype to default (float16)
            torch::autocast::set_autocast_ipu_dtype(torch::ScalarType::Half);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
