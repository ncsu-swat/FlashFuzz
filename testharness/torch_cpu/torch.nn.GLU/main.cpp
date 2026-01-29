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
        
        // GLU requires non-empty tensor with at least 1 dimension
        if (input.dim() == 0 || input.numel() == 0) {
            return 0;
        }
        
        // Extract a dimension value from the remaining data if available
        int64_t dim = -1; // Default dimension for GLU
        if (offset + sizeof(uint8_t) <= Size) {
            uint8_t dim_byte = Data[offset];
            offset += sizeof(uint8_t);
            // Map to valid dimension range [-input.dim(), input.dim()-1]
            dim = (dim_byte % input.dim());
        }
        
        // GLU requires the size at dim to be divisible by 2
        // Check if the dimension has even size, if not skip this input
        int64_t actual_dim = dim < 0 ? dim + input.dim() : dim;
        if (actual_dim < 0 || actual_dim >= input.dim()) {
            actual_dim = input.dim() - 1; // Default to last dimension
            dim = actual_dim;
        }
        
        if (input.size(actual_dim) < 2 || input.size(actual_dim) % 2 != 0) {
            // Try to reshape or create a tensor with even size along dim
            // For simplicity, we'll try to find a dimension with even size
            bool found_valid_dim = false;
            for (int64_t d = 0; d < input.dim(); d++) {
                if (input.size(d) >= 2 && input.size(d) % 2 == 0) {
                    dim = d;
                    found_valid_dim = true;
                    break;
                }
            }
            if (!found_valid_dim) {
                // No valid dimension found, skip this input
                return 0;
            }
        }
        
        // Create GLU module with the selected dimension
        torch::nn::GLUOptions options;
        options.dim(dim);
        
        auto glu = torch::nn::GLU(options);
        
        // Apply GLU to the input tensor
        torch::Tensor output;
        try {
            output = glu->forward(input);
        } catch (const c10::Error &e) {
            // Expected errors due to tensor constraints
            return 0;
        }
        
        // Ensure the output is valid by accessing some property
        auto output_sizes = output.sizes();
        
        // Try to access elements to ensure no segfaults
        if (output.numel() > 0) {
            auto first_element = output.flatten()[0].item<float>();
            (void)first_element; // Prevent unused variable warning
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}