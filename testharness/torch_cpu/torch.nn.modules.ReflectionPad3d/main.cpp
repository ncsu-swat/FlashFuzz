#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Ensure we have a 5D tensor (batch, channels, depth, height, width)
        // ReflectionPad3d requires 4D (unbatched) or 5D (batched) input
        if (input.dim() < 4) {
            std::vector<int64_t> new_shape = input.sizes().vec();
            while (new_shape.size() < 5) {
                new_shape.insert(new_shape.begin(), 1);
            }
            input = input.reshape(new_shape);
        } else if (input.dim() > 5) {
            // Collapse extra dimensions into last dimension
            int64_t batch = input.size(0);
            int64_t channels = input.size(1);
            int64_t depth = input.size(2);
            int64_t height = input.size(3);
            int64_t width = 1;
            for (int i = 4; i < input.dim(); i++) {
                width *= input.size(i);
            }
            input = input.reshape({batch, channels, depth, height, width});
        }
        
        // Ensure minimum size for reflection padding (each dim must be > padding)
        auto sizes = input.sizes().vec();
        for (int i = 2; i < 5; i++) {
            if (sizes[i] < 2) {
                sizes[i] = 2;
            }
        }
        input = input.reshape(sizes).contiguous();
        
        // Parse padding values from the remaining data
        // Padding format: (left, right, top, bottom, front, back)
        // Affects dimensions: (width, height, depth)
        std::vector<int64_t> padding(6, 0);
        for (int i = 0; i < 6 && offset + 1 <= Size; i++) {
            uint8_t pad_byte = Data[offset];
            offset++;
            // Ensure padding is less than corresponding dimension size for reflection
            int64_t dim_size;
            if (i < 2) {
                dim_size = input.size(4); // width
            } else if (i < 4) {
                dim_size = input.size(3); // height
            } else {
                dim_size = input.size(2); // depth
            }
            // Reflection padding requires padding < dim_size
            padding[i] = (pad_byte % std::max(dim_size - 1, (int64_t)1));
        }
        
        // Create ReflectionPad3d module with 6-element padding
        try {
            torch::nn::ReflectionPad3d reflection_pad(
                torch::nn::ReflectionPad3dOptions(
                    {padding[0], padding[1], padding[2], 
                     padding[3], padding[4], padding[5]}
                )
            );
            
            // Apply padding
            torch::Tensor output = reflection_pad->forward(input);
            
            // Verify output shape
            auto input_sizes = input.sizes();
            auto output_sizes = output.sizes();
            
            // Check that output has expected shape
            // Padding order: (left, right, top, bottom, front, back)
            // Dimensions: width (dim 4), height (dim 3), depth (dim 2)
            if (output_sizes[0] != input_sizes[0] || // batch
                output_sizes[1] != input_sizes[1] || // channels
                output_sizes[2] != input_sizes[2] + padding[4] + padding[5] || // depth
                output_sizes[3] != input_sizes[3] + padding[2] + padding[3] || // height
                output_sizes[4] != input_sizes[4] + padding[0] + padding[1]) { // width
                
                throw std::runtime_error("Output shape mismatch");
            }
        } catch (const c10::Error&) {
            // Expected failures for invalid padding configurations
        }
        
        // Try with single padding value for all sides
        if (offset + 1 <= Size) {
            uint8_t single_pad_byte = Data[offset];
            offset++;
            
            // Use a single padding value, must be less than all spatial dims
            int64_t min_dim = std::min({input.size(2), input.size(3), input.size(4)});
            int64_t single_pad = single_pad_byte % std::max(min_dim - 1, (int64_t)1);
            
            try {
                torch::nn::ReflectionPad3d reflection_pad_single{
                    torch::nn::ReflectionPad3dOptions(single_pad)
                };
                
                torch::Tensor output_single = reflection_pad_single->forward(input);
            } catch (const c10::Error&) {
                // Expected failures
            }
        }
        
        // Try with functional interface
        try {
            torch::Tensor output_functional = torch::nn::functional::pad(
                input,
                torch::nn::functional::PadFuncOptions(
                    {padding[0], padding[1], padding[2], 
                     padding[3], padding[4], padding[5]})
                    .mode(torch::kReflect)
            );
        } catch (const c10::Error&) {
            // Expected failures
        }
        
        // Test with 4D input (unbatched)
        if (input.size(0) == 1) {
            try {
                torch::Tensor input_4d = input.squeeze(0);
                torch::nn::ReflectionPad3d reflection_pad_4d(
                    torch::nn::ReflectionPad3dOptions(
                        {padding[0], padding[1], padding[2], 
                         padding[3], padding[4], padding[5]}
                    )
                );
                torch::Tensor output_4d = reflection_pad_4d->forward(input_4d);
            } catch (const c10::Error&) {
                // Expected failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}