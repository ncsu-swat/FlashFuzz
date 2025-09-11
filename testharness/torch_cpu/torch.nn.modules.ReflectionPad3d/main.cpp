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
        
        // Ensure we have a 5D tensor (batch, channels, depth, height, width)
        // ReflectionPad3d requires 5D input
        if (input.dim() != 5) {
            // Reshape to 5D if needed
            std::vector<int64_t> new_shape;
            if (input.dim() < 5) {
                // Add dimensions to make it 5D
                new_shape = input.sizes().vec();
                while (new_shape.size() < 5) {
                    new_shape.insert(new_shape.begin(), 1);
                }
            } else if (input.dim() > 5) {
                // Collapse extra dimensions
                new_shape.push_back(input.size(0)); // batch
                new_shape.push_back(input.size(1)); // channels
                
                int64_t depth = 1;
                int64_t height = 1;
                int64_t width = 1;
                
                if (input.dim() > 2) depth = input.size(2);
                if (input.dim() > 3) height = input.size(3);
                if (input.dim() > 4) {
                    // Collapse remaining dimensions into width
                    width = 1;
                    for (int i = 4; i < input.dim(); i++) {
                        width *= input.size(i);
                    }
                }
                
                new_shape.push_back(depth);
                new_shape.push_back(height);
                new_shape.push_back(width);
            }
            
            // Reshape the tensor
            input = input.reshape(new_shape);
        }
        
        // Parse padding values from the remaining data
        std::vector<int64_t> padding(6, 0);
        for (int i = 0; i < 6 && offset + sizeof(int64_t) <= Size; i++) {
            int64_t pad_value;
            std::memcpy(&pad_value, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure padding is not too large to avoid excessive memory usage
            padding[i] = std::abs(pad_value) % 10;
        }
        
        // Create ReflectionPad3d module
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
        if (output_sizes[0] != input_sizes[0] || // batch
            output_sizes[1] != input_sizes[1] || // channels
            output_sizes[2] != input_sizes[2] + padding[0] + padding[1] || // depth
            output_sizes[3] != input_sizes[3] + padding[2] + padding[3] || // height
            output_sizes[4] != input_sizes[4] + padding[4] + padding[5]) { // width
            
            throw std::runtime_error("Output shape mismatch");
        }
        
        // Try with different padding values
        if (offset + sizeof(int64_t) <= Size) {
            int64_t alt_pad_value;
            std::memcpy(&alt_pad_value, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Use a single padding value for all dimensions
            int64_t single_pad = std::abs(alt_pad_value) % 5;
            
            torch::nn::ReflectionPad3d reflection_pad_single{
                torch::nn::ReflectionPad3dOptions(single_pad)
            };
            
            torch::Tensor output_single = reflection_pad_single->forward(input);
        }
        
        // Try with functional interface if available
        if (input.dim() == 5) {
            torch::Tensor output_functional = torch::nn::functional::pad(
                input,
                torch::nn::functional::PadFuncOptions({padding[0], padding[1], padding[2], 
                                                      padding[3], padding[4], padding[5]})
                    .mode(torch::kReflect)
            );
        }
    }
    catch (const std::exception &e)
    {
        return 0; // keep the input
    }
    return 0; // keep the input
}
