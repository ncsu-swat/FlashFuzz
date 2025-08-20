#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract padding values from the remaining data
        std::vector<int64_t> padding;
        for (int i = 0; i < 4 && offset + sizeof(int64_t) <= Size; i++) {
            int64_t pad_value;
            std::memcpy(&pad_value, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // No need to restrict padding to positive values - let the API handle it
            padding.push_back(pad_value);
        }
        
        // If we don't have enough data for 4 padding values, use defaults
        while (padding.size() < 4) {
            // Use the next byte as padding if available
            if (offset < Size) {
                padding.push_back(static_cast<int64_t>(Data[offset++]));
            } else {
                padding.push_back(1); // Default padding
            }
        }
        
        // Create ReflectionPad2d module with different configurations
        torch::nn::ReflectionPad2d pad_module = nullptr;
        
        // Use a byte to determine which constructor to use
        if (offset < Size) {
            uint8_t constructor_choice = Data[offset++];
            
            if (constructor_choice % 2 == 0) {
                // Use the constructor with a single padding value
                int64_t single_pad = padding[0];
                pad_module = torch::nn::ReflectionPad2d(single_pad);
            } else {
                // Use the constructor with separate padding values
                pad_module = torch::nn::ReflectionPad2d(
                    torch::nn::ReflectionPad2dOptions(padding));
            }
        } else {
            // Default to using all four padding values
            pad_module = torch::nn::ReflectionPad2d(
                torch::nn::ReflectionPad2dOptions(padding));
        }
        
        // Apply the padding operation
        torch::Tensor output = pad_module->forward(input);
        
        // Access some properties of the output tensor to ensure it's computed
        auto output_sizes = output.sizes();
        auto output_dtype = output.dtype();
        
        // Try to access the first element if tensor is not empty
        if (output.numel() > 0) {
            auto first_element = output.flatten()[0];
        }
        
        // Try to perform additional operations on the output
        if (output.numel() > 0) {
            torch::Tensor squared = output * output;
            torch::Tensor summed = torch::sum(output);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}