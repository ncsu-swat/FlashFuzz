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
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a 5D tensor (batch, channels, depth, height, width)
        // If not, reshape it to make it compatible with ZeroPad3d
        if (input_tensor.dim() < 5) {
            std::vector<int64_t> new_shape;
            
            // Keep original dimensions
            for (int i = 0; i < input_tensor.dim(); i++) {
                new_shape.push_back(input_tensor.size(i));
            }
            
            // Add missing dimensions
            while (new_shape.size() < 5) {
                new_shape.push_back(1);
            }
            
            input_tensor = input_tensor.reshape(new_shape);
        }
        
        // Extract padding values from the input data
        int64_t padding_left = 0;
        int64_t padding_right = 0;
        int64_t padding_top = 0;
        int64_t padding_bottom = 0;
        int64_t padding_front = 0;
        int64_t padding_back = 0;
        
        // Parse padding values if we have enough data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_left, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_right, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_top, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_bottom, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_front, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&padding_back, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Try different padding configurations
        try {
            // Case 1: Single integer for all sides
            if (offset < Size) {
                int64_t single_padding = 0;
                std::memcpy(&single_padding, Data + offset, sizeof(int64_t));
                auto pad_module1 = torch::nn::ZeroPad3d(single_padding);
                auto output1 = pad_module1->forward(input_tensor);
            }
        } catch (const std::exception &) {
            // Ignore exceptions and continue
        }
        
        try {
            // Case 2: Tuple of 6 values (left, right, top, bottom, front, back)
            auto pad_module2 = torch::nn::ZeroPad3d(torch::nn::ZeroPad3dOptions(
                {padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back}));
            auto output2 = pad_module2->forward(input_tensor);
        } catch (const std::exception &) {
            // Ignore exceptions and continue
        }
        
        try {
            // Case 3: Using functional interface
            auto output3 = torch::nn::functional::pad(
                input_tensor,
                torch::nn::functional::PadFuncOptions({padding_front, padding_back, padding_top, padding_bottom, padding_left, padding_right})
                    .mode(torch::kConstant)
                    .value(0.0));
        } catch (const std::exception &) {
            // Ignore exceptions and continue
        }
        
        // Edge case: Try with negative padding values
        try {
            auto pad_module4 = torch::nn::ZeroPad3d(torch::nn::ZeroPad3dOptions(
                {-padding_left, -padding_right, -padding_top, -padding_bottom, -padding_front, -padding_back}));
            auto output4 = pad_module4->forward(input_tensor);
        } catch (const std::exception &) {
            // Ignore exceptions and continue
        }
        
        // Edge case: Try with very large padding values
        try {
            int64_t large_padding = std::numeric_limits<int16_t>::max();
            auto pad_module5 = torch::nn::ZeroPad3d(large_padding);
            auto output5 = pad_module5->forward(input_tensor);
        } catch (const std::exception &) {
            // Ignore exceptions and continue
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
