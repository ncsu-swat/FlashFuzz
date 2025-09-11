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
        
        // Need at least a few bytes for the input tensor and padding values
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 2D tensor for ReflectionPad2d
        if (input.dim() < 2) {
            if (input.dim() == 0) {
                input = input.unsqueeze(0).unsqueeze(0);
            } else if (input.dim() == 1) {
                input = input.unsqueeze(0);
            }
        }
        
        // Extract padding values from the remaining data
        int64_t padding[4] = {1, 1, 1, 1}; // Default padding
        
        for (int i = 0; i < 4 && offset + sizeof(int64_t) <= Size; i++) {
            int64_t pad_value;
            std::memcpy(&pad_value, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Use modulo to keep padding values within reasonable range
            // Allow negative values to test error handling
            padding[i] = pad_value % 10;
        }
        
        // Create ReflectionPad2d module with different padding configurations
        
        // 1. Single value padding (same for all sides)
        {
            auto pad_module1 = torch::nn::ReflectionPad2d(padding[0]);
            auto output1 = pad_module1->forward(input);
        }
        
        // 2. Four-value padding (left, right, top, bottom)
        {
            auto pad_module2 = torch::nn::ReflectionPad2d(
                torch::nn::ReflectionPad2dOptions(
                    {padding[0], padding[1], padding[2], padding[3]}
                )
            );
            auto output2 = pad_module2->forward(input);
        }
        
        // 3. Functional interface
        {
            auto output3 = torch::nn::functional::pad(
                input,
                torch::nn::functional::PadFuncOptions({padding[0], padding[1], padding[2], padding[3]})
                    .mode(torch::kReflect)
            );
        }
        
        // 4. Edge case: zero padding
        {
            auto pad_module4 = torch::nn::ReflectionPad2d(0);
            auto output4 = pad_module4->forward(input);
        }
        
        // 5. Try with different tensor types
        if (offset + 1 <= Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            // Convert input tensor to the selected data type
            torch::Tensor converted_input = input.to(dtype);
            
            auto pad_module5 = torch::nn::ReflectionPad2d(
                torch::nn::ReflectionPad2dOptions({padding[0], padding[1], padding[2], padding[3]})
            );
            
            auto output5 = pad_module5->forward(converted_input);
        }
        
        // 6. Try with non-contiguous tensor
        if (input.dim() >= 3 && input.size(0) > 1) {
            torch::Tensor non_contiguous = input.transpose(0, 1);
            
            // Ensure it's actually non-contiguous
            if (!non_contiguous.is_contiguous()) {
                auto pad_module6 = torch::nn::ReflectionPad2d(
                    torch::nn::ReflectionPad2dOptions({padding[0], padding[1], padding[2], padding[3]})
                );
                
                auto output6 = pad_module6->forward(non_contiguous);
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
