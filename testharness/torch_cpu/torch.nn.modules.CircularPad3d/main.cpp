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
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 6 bytes left for padding values
        if (Size - offset < 6) {
            return 0;
        }
        
        // Extract padding values from the input data
        std::vector<int64_t> padding;
        for (int i = 0; i < 6; i++) {
            if (offset < Size) {
                // Use modulo to keep padding values within a reasonable range
                padding.push_back(static_cast<int64_t>(Data[offset++]) % 10 - 5);
            } else {
                padding.push_back(0);
            }
        }
        
        // Apply padding to the input tensor
        torch::Tensor output;
        
        // CircularPad3d expects a 5D tensor (batch, channels, depth, height, width)
        // If the tensor doesn't have 5 dimensions, we'll try to adapt it
        if (input_tensor.dim() != 5) {
            // For tensors with fewer dimensions, expand to 5D
            if (input_tensor.dim() < 5) {
                std::vector<int64_t> new_shape;
                for (int i = 0; i < 5 - input_tensor.dim(); i++) {
                    new_shape.push_back(1);
                }
                for (int i = 0; i < input_tensor.dim(); i++) {
                    new_shape.push_back(input_tensor.size(i));
                }
                input_tensor = input_tensor.reshape(new_shape);
            }
            // For tensors with more dimensions, collapse extra dimensions
            else if (input_tensor.dim() > 5) {
                std::vector<int64_t> new_shape;
                int64_t collapsed_size = 1;
                for (int i = 0; i < input_tensor.dim() - 4; i++) {
                    collapsed_size *= input_tensor.size(i);
                }
                new_shape.push_back(collapsed_size);
                for (int i = input_tensor.dim() - 4; i < input_tensor.dim(); i++) {
                    new_shape.push_back(input_tensor.size(i));
                }
                input_tensor = input_tensor.reshape(new_shape);
            }
        }
        
        // Apply the circular padding operation using functional interface
        output = torch::nn::functional::pad(input_tensor, padding, torch::nn::functional::PadFuncOptions().mode(torch::kCircular));
        
        // Verify the output has the expected shape
        auto input_sizes = input_tensor.sizes();
        auto output_sizes = output.sizes();
        
        // Check that batch and channel dimensions are preserved
        assert(output_sizes[0] == input_sizes[0]);
        assert(output_sizes[1] == input_sizes[1]);
        
        // Check that spatial dimensions are padded correctly
        assert(output_sizes[2] == input_sizes[2] + padding[4] + padding[5]);
        assert(output_sizes[3] == input_sizes[3] + padding[2] + padding[3]);
        assert(output_sizes[4] == input_sizes[4] + padding[0] + padding[1]);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}