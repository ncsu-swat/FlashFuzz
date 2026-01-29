#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // diagflat works on 1D input or flattens the input
        // Flatten if not already 1D to test the flattening behavior
        if (input.dim() > 1 && offset < Size && (Data[offset] % 2 == 0)) {
            // Sometimes test with already flattened input
            input = input.flatten();
        }
        
        // Parse offset parameter (diagonal offset)
        int64_t diag_offset = 0;
        if (offset + sizeof(int8_t) <= Size) {
            // Use int8_t to get reasonable offset values (-128 to 127)
            int8_t small_offset;
            std::memcpy(&small_offset, Data + offset, sizeof(int8_t));
            diag_offset = static_cast<int64_t>(small_offset);
            offset += sizeof(int8_t);
        }
        
        // Apply diagflat operation
        torch::Tensor result;
        
        // Try different variants based on fuzzer data
        bool use_offset = (offset < Size) && (Data[offset % Size] % 2 == 0);
        
        if (use_offset) {
            // Use the diagonal offset parameter
            result = torch::diagflat(input, diag_offset);
        } else {
            // Default offset = 0
            result = torch::diagflat(input);
        }
        
        // Verify the result is a valid tensor
        if (!result.defined()) {
            throw std::runtime_error("diagflat returned undefined tensor");
        }
        
        // Access properties to ensure the tensor is valid
        auto sizes = result.sizes();
        auto dtype = result.dtype();
        
        // Verify the result is 2D (diagflat always produces 2D output)
        if (result.dim() != 2) {
            throw std::runtime_error("diagflat should produce 2D output");
        }
        
        // Try to perform some operations on the result
        if (result.numel() > 0) {
            auto sum = torch::sum(result);
            
            // Mean only works on floating point tensors
            if (result.is_floating_point()) {
                auto mean = torch::mean(result);
            }
            
            // Extract diagonal to verify roundtrip
            auto diag = torch::diag(result, diag_offset);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}