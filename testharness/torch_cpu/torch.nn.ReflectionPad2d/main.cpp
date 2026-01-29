#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for the input tensor and padding values
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // ReflectionPad2d requires 3D (C, H, W) or 4D (N, C, H, W) input
        while (input.dim() < 3) {
            input = input.unsqueeze(0);
        }
        
        // Ensure minimum spatial dimensions for reflection padding
        // ReflectionPad2d needs H and W > 0
        if (input.size(-1) < 1 || input.size(-2) < 1) {
            return 0;
        }
        
        // Extract padding values from the remaining data
        int64_t padding[4] = {0, 0, 0, 0}; // Default padding
        
        for (int i = 0; i < 4 && offset + sizeof(int8_t) <= Size; i++) {
            int8_t pad_value = static_cast<int8_t>(Data[offset++]);
            // Keep padding values small and non-negative for valid cases
            // Use absolute value and modulo to get reasonable padding
            padding[i] = std::abs(pad_value) % 5;
        }
        
        // 1. Single value padding (same for all sides)
        try {
            auto pad_module1 = torch::nn::ReflectionPad2d(
                torch::nn::ReflectionPad2dOptions(padding[0])
            );
            auto output1 = pad_module1->forward(input);
        } catch (...) {
            // Silently catch - padding might exceed tensor dimensions
        }
        
        // 2. Four-value padding (left, right, top, bottom)
        try {
            auto pad_module2 = torch::nn::ReflectionPad2d(
                torch::nn::ReflectionPad2dOptions(
                    {padding[0], padding[1], padding[2], padding[3]}
                )
            );
            auto output2 = pad_module2->forward(input);
        } catch (...) {
            // Silently catch - invalid padding combinations
        }
        
        // 3. Functional interface
        try {
            auto output3 = torch::nn::functional::pad(
                input,
                torch::nn::functional::PadFuncOptions({padding[0], padding[1], padding[2], padding[3]})
                    .mode(torch::kReflect)
            );
        } catch (...) {
            // Silently catch
        }
        
        // 4. Edge case: zero padding
        try {
            auto pad_module4 = torch::nn::ReflectionPad2d(
                torch::nn::ReflectionPad2dOptions(0)
            );
            auto output4 = pad_module4->forward(input);
        } catch (...) {
            // Silently catch
        }
        
        // 5. Try with different tensor types
        if (offset + 1 <= Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            try {
                // Convert input tensor to the selected data type
                torch::Tensor converted_input = input.to(dtype);
                
                auto pad_module5 = torch::nn::ReflectionPad2d(
                    torch::nn::ReflectionPad2dOptions({padding[0], padding[1], padding[2], padding[3]})
                );
                
                auto output5 = pad_module5->forward(converted_input);
            } catch (...) {
                // Silently catch - some dtypes may not be supported
            }
        }
        
        // 6. Try with non-contiguous tensor
        if (input.dim() >= 4 && input.size(0) > 1) {
            try {
                torch::Tensor non_contiguous = input.transpose(0, 1);
                
                if (!non_contiguous.is_contiguous()) {
                    auto pad_module6 = torch::nn::ReflectionPad2d(
                        torch::nn::ReflectionPad2dOptions({padding[0], padding[1], padding[2], padding[3]})
                    );
                    
                    auto output6 = pad_module6->forward(non_contiguous);
                }
            } catch (...) {
                // Silently catch
            }
        }
        
        // 7. Test with 4D input explicitly (batch mode)
        if (input.dim() == 3) {
            try {
                torch::Tensor batched_input = input.unsqueeze(0);
                auto pad_module7 = torch::nn::ReflectionPad2d(
                    torch::nn::ReflectionPad2dOptions({padding[0], padding[1], padding[2], padding[3]})
                );
                auto output7 = pad_module7->forward(batched_input);
            } catch (...) {
                // Silently catch
            }
        }
        
        // 8. Test asymmetric padding
        try {
            auto pad_module8 = torch::nn::ReflectionPad2d(
                torch::nn::ReflectionPad2dOptions({padding[0], padding[1] + 1, padding[2], padding[3] + 1})
            );
            auto output8 = pad_module8->forward(input);
        } catch (...) {
            // Silently catch
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}