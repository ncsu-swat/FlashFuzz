#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for diff operation if we have more data
        int64_t n = 1;  // Default value
        int64_t dim = 0; // Default value
        bool use_prepend = false;
        bool use_append = false;
        
        // Parse n parameter if we have data
        if (offset + sizeof(int64_t) <= Size) {
            int64_t raw_n;
            std::memcpy(&raw_n, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Allow negative values to test error cases
            n = raw_n;
        }
        
        // Parse dim parameter if we have data
        if (offset + sizeof(int64_t) <= Size) {
            int64_t raw_dim;
            std::memcpy(&raw_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Allow negative values to test error cases
            dim = raw_dim;
        }
        
        // Parse prepend flag if we have data
        if (offset < Size) {
            use_prepend = Data[offset++] & 0x1;
        }
        
        // Parse append flag if we have data
        if (offset < Size) {
            use_append = Data[offset++] & 0x1;
        }
        
        // Apply the diff operation
        torch::Tensor result;
        
        // Try different variants of the diff operation
        if (offset % 4 == 0) {
            // Variant 1: Just n
            result = torch::diff(input, n);
        } else if (offset % 4 == 1) {
            // Variant 2: n and dim
            result = torch::diff(input, n, dim);
        } else if (offset % 4 == 2) {
            // Variant 3: n, dim, prepend
            std::optional<torch::Tensor> prepend_tensor;
            if (use_prepend) {
                prepend_tensor = torch::ones({1});
            }
            result = torch::diff(input, n, dim, prepend_tensor);
        } else {
            // Variant 4: n, dim, prepend, append
            std::optional<torch::Tensor> prepend_tensor;
            std::optional<torch::Tensor> append_tensor;
            if (use_prepend) {
                prepend_tensor = torch::ones({1});
            }
            if (use_append) {
                append_tensor = torch::zeros({1});
            }
            result = torch::diff(input, n, dim, prepend_tensor, append_tensor);
        }
        
        // Perform some operation on the result to ensure it's used
        auto sum = result.sum();
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}