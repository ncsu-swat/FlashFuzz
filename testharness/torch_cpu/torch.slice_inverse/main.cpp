#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <algorithm>      // For std::max

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
        
        // Extract parameters for slice_inverse
        // We need: input, dim, start, end, step
        
        // Get dim parameter (can be negative)
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Get start parameter (can be negative)
        int64_t start = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&start, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Get end parameter (can be negative)
        int64_t end = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&end, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Get step parameter (should be non-zero)
        int64_t step = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&step, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            if (step == 0) step = 1; // Avoid division by zero
        }
        
        // Create a values tensor to be inserted
        torch::Tensor values;
        if (offset < Size) {
            values = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a simple tensor
            values = torch::ones_like(input);
        }
        
        // Apply slice_inverse operation
        torch::Tensor result;
        
        // Handle potential edge cases
        if (input.dim() > 0) {
            // Normalize dim to be within valid range
            if (dim < 0) {
                dim = input.dim() + dim;
            }
            dim = dim % std::max(static_cast<int64_t>(1), input.dim());
            
            // Apply slice_inverse with optional parameters
            std::optional<int64_t> start_opt = (start != 0) ? std::optional<int64_t>(start) : std::nullopt;
            std::optional<int64_t> end_opt = (end != 0) ? std::optional<int64_t>(end) : std::nullopt;
            
            result = torch::slice_inverse(input, values, dim, start_opt, end_opt, step);
        } else {
            // For scalar tensors, just return the input as slice_inverse may not be applicable
            result = input;
        }
        
        // Ensure the result is valid
        if (result.defined()) {
            // Access some elements to ensure computation is done
            if (result.numel() > 0) {
                auto sum = result.sum().item<float>();
                (void)sum; // Prevent unused variable warning
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
