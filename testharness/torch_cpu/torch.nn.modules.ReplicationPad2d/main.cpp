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
        
        // Ensure we have at least 2D tensor for ReplicationPad2d
        if (input.dim() < 2) {
            // Reshape to at least 2D if needed
            std::vector<int64_t> new_shape;
            if (input.dim() == 0) {
                // Scalar tensor, reshape to 2D
                new_shape = {1, 1};
            } else if (input.dim() == 1) {
                // 1D tensor, reshape to 2D
                new_shape = {1, input.size(0)};
            }
            input = input.reshape(new_shape);
        }
        
        // Extract padding values from the remaining data
        int64_t padding[4] = {1, 1, 1, 1}; // Default padding
        
        for (int i = 0; i < 4 && offset + sizeof(int64_t) <= Size; i++) {
            int64_t pad_value;
            std::memcpy(&pad_value, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Use the raw value without bounds checking to test edge cases
            padding[i] = pad_value;
        }
        
        // Create ReplicationPad2d module
        torch::nn::ReplicationPad2d pad_module(
            torch::nn::ReplicationPad2dOptions({padding[0], padding[1], padding[2], padding[3]})
        );
        
        // Apply padding
        torch::Tensor output = pad_module->forward(input);
        
        // Try different padding configurations if we have more data
        if (offset + sizeof(int64_t) <= Size) {
            int64_t single_pad;
            std::memcpy(&single_pad, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Try with a single padding value
            auto pad_module2 = torch::nn::ReplicationPad2d(
                torch::nn::ReplicationPad2dOptions(single_pad)
            );
            torch::Tensor output2 = pad_module2->forward(input);
        }
        
        // Try with different tensor types
        if (offset < Size) {
            auto dtype = fuzzer_utils::parseDataType(Data[offset++]);
            if (dtype != input.dtype()) {
                try {
                    torch::Tensor input_cast = input.to(dtype);
                    torch::Tensor output_cast = pad_module->forward(input_cast);
                } catch (...) {
                    // Ignore errors from type conversion
                }
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