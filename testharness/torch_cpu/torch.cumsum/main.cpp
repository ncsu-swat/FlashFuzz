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
        
        // Need at least a few bytes to create a tensor and specify dim
        if (Size < 3) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get a dimension to perform cumsum along
        int64_t dim = 0;
        if (offset < Size) {
            // Extract a dimension from the input data
            int64_t raw_dim;
            size_t bytes_to_read = std::min(sizeof(int64_t), Size - offset);
            std::memcpy(&raw_dim, Data + offset, bytes_to_read);
            offset += bytes_to_read;
            
            // If tensor has dimensions, select one of them
            if (input.dim() > 0) {
                // Use modulo to ensure dim is within valid range
                // Allow negative dimensions for testing negative indexing
                dim = raw_dim % (2 * input.dim()) - input.dim();
            }
        }
        
        // Get dtype for output (optional)
        torch::ScalarType dtype = input.scalar_type();
        if (offset < Size) {
            dtype = fuzzer_utils::parseDataType(Data[offset++]);
        }
        
        // Apply cumsum operation
        torch::Tensor output;
        
        // Test different variants of cumsum
        if (offset < Size && Data[offset] % 3 == 0) {
            // Variant 1: cumsum with dimension only
            output = torch::cumsum(input, dim);
        } 
        else if (offset < Size && Data[offset] % 3 == 1) {
            // Variant 2: cumsum with dimension and dtype
            output = torch::cumsum(input, dim, dtype);
        }
        else {
            // Variant 3: cumsum using Tensor method
            output = input.cumsum(dim);
        }
        
        // Perform a simple operation on the result to ensure it's used
        auto sum = output.sum();
        
        // Test in-place version if we have more data
        if (offset < Size && Data[offset] % 2 == 0) {
            torch::Tensor input_copy = input.clone();
            input_copy.cumsum_(dim);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
