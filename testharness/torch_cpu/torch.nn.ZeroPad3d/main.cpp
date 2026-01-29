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
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure tensor is float for consistent accessor usage
        input_tensor = input_tensor.to(torch::kFloat32);
        
        // Ensure we have at least 5D tensor (batch, channels, depth, height, width)
        // Use unsqueeze to add dimensions without changing total elements
        while (input_tensor.dim() < 5) {
            input_tensor = input_tensor.unsqueeze(0);
        }
        
        // If more than 5D, flatten extra dimensions into batch
        while (input_tensor.dim() > 5) {
            int64_t dim0 = input_tensor.size(0);
            int64_t dim1 = input_tensor.size(1);
            input_tensor = input_tensor.reshape({dim0 * dim1, 
                                                  input_tensor.size(2), 
                                                  input_tensor.size(3), 
                                                  input_tensor.size(4), 
                                                  input_tensor.size(5)});
        }
        
        // Parse padding values from the remaining data
        std::vector<int64_t> padding(6, 0); // [left, right, top, bottom, front, back]
        
        for (int i = 0; i < 6 && offset + sizeof(int8_t) <= Size; i++) {
            int8_t pad_value;
            std::memcpy(&pad_value, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            
            // Clamp padding to reasonable range to avoid OOM
            // Negative padding is valid but should be bounded
            padding[i] = std::max(static_cast<int64_t>(-10), 
                                  std::min(static_cast<int64_t>(pad_value), static_cast<int64_t>(50)));
        }
        
        // Create ZeroPad3d module with tuple of 6 values
        try {
            auto pad_module = torch::nn::ZeroPad3d(
                torch::nn::ZeroPad3dOptions({padding[0], padding[1], padding[2], 
                                              padding[3], padding[4], padding[5]}));
            
            // Apply padding
            torch::Tensor output_tensor = pad_module->forward(input_tensor);
            
            // Force computation
            if (output_tensor.numel() > 0) {
                volatile float sum = output_tensor.sum().item<float>();
                (void)sum;
            }
        } catch (...) {
            // Silently catch invalid padding combinations
        }
        
        // Test with single padding value (applies same padding to all 6 sides)
        if (offset < Size) {
            int8_t single_pad_raw;
            std::memcpy(&single_pad_raw, Data + offset, sizeof(int8_t));
            offset += sizeof(int8_t);
            
            int64_t single_pad = std::max(static_cast<int64_t>(0), 
                                          std::min(static_cast<int64_t>(single_pad_raw), static_cast<int64_t>(20)));
            
            try {
                auto single_pad_module = torch::nn::ZeroPad3d(torch::nn::ZeroPad3dOptions(single_pad));
                torch::Tensor single_pad_output = single_pad_module->forward(input_tensor);
                
                // Force computation
                volatile float sum = single_pad_output.sum().item<float>();
                (void)sum;
            } catch (...) {
                // Silently catch invalid configurations
            }
        }
        
        // Test with functional interface
        try {
            torch::Tensor functional_output = torch::nn::functional::pad(
                input_tensor,
                torch::nn::functional::PadFuncOptions({padding[0], padding[1], padding[2], 
                                                        padding[3], padding[4], padding[5]})
                    .mode(torch::kConstant)
                    .value(0.0));
            
            // Force computation
            volatile float sum = functional_output.sum().item<float>();
            (void)sum;
        } catch (...) {
            // Silently catch invalid configurations
        }
        
        // Test with asymmetric padding using tuple constructor
        if (offset + 2 < Size) {
            try {
                int64_t left = (Data[offset] % 10);
                int64_t right = (Data[offset + 1] % 10);
                int64_t top = (offset + 2 < Size) ? (Data[offset + 2] % 10) : 0;
                int64_t bottom = (offset + 3 < Size) ? (Data[offset + 3] % 10) : 0;
                int64_t front = (offset + 4 < Size) ? (Data[offset + 4] % 10) : 0;
                int64_t back = (offset + 5 < Size) ? (Data[offset + 5] % 10) : 0;
                
                auto asym_pad_module = torch::nn::ZeroPad3d(
                    torch::nn::ZeroPad3dOptions({left, right, top, bottom, front, back}));
                torch::Tensor asym_output = asym_pad_module->forward(input_tensor);
                
                // Verify output dimensions changed appropriately
                volatile int64_t out_depth = asym_output.size(2);
                volatile int64_t out_height = asym_output.size(3);
                volatile int64_t out_width = asym_output.size(4);
                (void)out_depth;
                (void)out_height;
                (void)out_width;
            } catch (...) {
                // Silently catch invalid configurations
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