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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 1 dimension
        if (input.dim() == 0) {
            input = input.unsqueeze(0);
        }
        
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
            // Constrain n to reasonable values (1 to input size along dim)
            n = (std::abs(raw_n) % 10) + 1;
        }
        
        // Parse dim parameter if we have data
        if (offset + sizeof(int64_t) <= Size) {
            int64_t raw_dim;
            std::memcpy(&raw_dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // Constrain dim to valid range
            dim = raw_dim % input.dim();
        }
        
        // Parse prepend flag if we have data
        if (offset < Size) {
            use_prepend = Data[offset++] & 0x1;
        }
        
        // Parse append flag if we have data
        if (offset < Size) {
            use_append = Data[offset++] & 0x1;
        }
        
        // Parse variant selector
        uint8_t variant = 0;
        if (offset < Size) {
            variant = Data[offset++] % 4;
        }
        
        // Apply the diff operation
        torch::Tensor result;
        
        // Try different variants of the diff operation
        if (variant == 0) {
            // Variant 1: Just n
            result = torch::diff(input, n);
        } else if (variant == 1) {
            // Variant 2: n and dim
            result = torch::diff(input, n, dim);
        } else if (variant == 2) {
            // Variant 3: n, dim, prepend
            std::optional<torch::Tensor> prepend_tensor = std::nullopt;
            if (use_prepend) {
                // Create prepend tensor with matching shape except along dim
                auto sizes = input.sizes().vec();
                sizes[dim] = 1;
                prepend_tensor = torch::ones(sizes, input.options());
            }
            result = torch::diff(input, n, dim, prepend_tensor);
        } else {
            // Variant 4: n, dim, prepend, append
            std::optional<torch::Tensor> prepend_tensor = std::nullopt;
            std::optional<torch::Tensor> append_tensor = std::nullopt;
            if (use_prepend) {
                auto sizes = input.sizes().vec();
                sizes[dim] = 1;
                prepend_tensor = torch::ones(sizes, input.options());
            }
            if (use_append) {
                auto sizes = input.sizes().vec();
                sizes[dim] = 1;
                append_tensor = torch::zeros(sizes, input.options());
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