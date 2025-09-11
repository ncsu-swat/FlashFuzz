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
        
        // Create src tensor to scatter into the input
        torch::Tensor src;
        if (offset < Size) {
            src = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we've consumed all data, create a simple tensor
            src = torch::ones_like(input);
        }
        
        // Extract slice parameters from remaining data
        int64_t dim = 0;
        int64_t start = 0;
        int64_t end = 0;
        int64_t step = 1;
        
        // Get dimension to slice along
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // If tensor is not empty, ensure dim is within valid range
            if (input.dim() > 0) {
                dim = dim % input.dim();
                if (dim < 0) dim += input.dim();
            }
        }
        
        // Get start index
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&start, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Get end index
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&end, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Get step value
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&step, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Avoid step=0 which would cause division by zero
            if (step == 0) step = 1;
        }
        
        // Apply slice_scatter operation
        torch::Tensor result;
        try {
            result = torch::slice_scatter(input, src, dim, start, end, step);
        } catch (const c10::Error& e) {
            // PyTorch specific exceptions are expected and not a fuzzer error
            return 0;
        }
        
        // Verify the result is a valid tensor
        if (result.defined() && !result.isnan().any().item<bool>() && 
            !result.isinf().any().item<bool>()) {
            // Optional: perform additional checks on the result
            if (result.sizes() != input.sizes()) {
                // This shouldn't happen for slice_scatter, so report if it does
                std::cerr << "Unexpected shape change in slice_scatter result" << std::endl;
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
