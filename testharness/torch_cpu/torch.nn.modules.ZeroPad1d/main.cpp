#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

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
        // Need at least a few bytes for tensor creation and padding values
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Extract padding values from data (constrained to reasonable range)
        int8_t pad_left_raw = static_cast<int8_t>(Data[offset++]);
        int8_t pad_right_raw = static_cast<int8_t>(Data[offset++]);
        
        // Constrain padding to reasonable positive values (0-32)
        int64_t padding_left = std::abs(pad_left_raw) % 33;
        int64_t padding_right = std::abs(pad_right_raw) % 33;
        
        // Get dimensions for input tensor (ZeroPad1d expects 2D or 3D input)
        uint8_t dim_choice = Data[offset++] % 2;
        uint8_t width = (Data[offset++] % 64) + 1;  // 1-64
        
        torch::Tensor input;
        
        if (dim_choice == 0) {
            // 2D input: (C, W)
            uint8_t channels = (Data[offset++] % 16) + 1;  // 1-16
            input = torch::randn({channels, width});
        } else {
            // 3D input: (N, C, W)
            uint8_t batch = (Data[offset++] % 8) + 1;      // 1-8
            uint8_t channels = (Data[offset++] % 16) + 1;  // 1-16
            input = torch::randn({batch, channels, width});
        }

        // Case 1: Single integer padding (symmetric)
        {
            int64_t single_padding = padding_left;
            torch::nn::ZeroPad1d pad(single_padding);
            
            try {
                torch::Tensor output = pad->forward(input);
                // Verify output shape
                (void)output.sizes();
            } catch (const c10::Error&) {
                // Expected for invalid configurations
            }
        }

        // Case 2: Asymmetric padding with tuple
        {
            torch::nn::ZeroPad1d pad(torch::nn::ZeroPad1dOptions({padding_left, padding_right}));
            
            try {
                torch::Tensor output = pad->forward(input);
                (void)output.sizes();
            } catch (const c10::Error&) {
                // Expected for invalid configurations
            }
        }

        // Case 3: Test with different data types
        if (offset + 1 < Size) {
            uint8_t dtype_choice = Data[offset++] % 4;
            torch::Tensor typed_input;
            
            try {
                switch (dtype_choice) {
                    case 0:
                        typed_input = input.to(torch::kFloat32);
                        break;
                    case 1:
                        typed_input = input.to(torch::kFloat64);
                        break;
                    case 2:
                        typed_input = input.to(torch::kInt32);
                        break;
                    default:
                        typed_input = input.to(torch::kInt64);
                        break;
                }
                
                torch::nn::ZeroPad1d pad(torch::nn::ZeroPad1dOptions({padding_left, padding_right}));
                torch::Tensor output = pad->forward(typed_input);
                (void)output.sizes();
            } catch (const c10::Error&) {
                // Expected for unsupported dtypes
            }
        }

        // Case 4: Test with zero padding
        {
            torch::nn::ZeroPad1d pad(torch::nn::ZeroPad1dOptions({0, 0}));
            
            try {
                torch::Tensor output = pad->forward(input);
                (void)output.sizes();
            } catch (const c10::Error&) {
                // Expected for invalid configurations
            }
        }

        // Case 5: Test functional interface (torch::nn::functional::pad)
        {
            try {
                torch::Tensor output = torch::nn::functional::pad(
                    input, 
                    torch::nn::functional::PadFuncOptions({padding_left, padding_right}).mode(torch::kConstant).value(0)
                );
                (void)output.sizes();
            } catch (const c10::Error&) {
                // Expected for invalid configurations
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