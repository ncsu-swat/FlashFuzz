#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least enough data for tensor dimensions and upscale factor
        if (Size < 20) return 0;

        // Extract tensor dimensions
        int batch_size = extract_int(Data, Size, offset, 1, 8);
        int channels = extract_int(Data, Size, offset, 1, 64);
        int height = extract_int(Data, Size, offset, 1, 32);
        int width = extract_int(Data, Size, offset, 1, 32);
        
        // Extract upscale factor (1 to 8 to avoid extremely large tensors)
        int upscale_factor = extract_int(Data, Size, offset, 1, 8);
        
        // For pixel_shuffle, the input channels must be divisible by upscale_factor^2
        // Adjust channels to satisfy this constraint
        int required_factor = upscale_factor * upscale_factor;
        channels = ((channels + required_factor - 1) / required_factor) * required_factor;
        
        // Ensure we have reasonable tensor sizes to avoid OOM
        if (batch_size * channels * height * width > 100000) return 0;
        
        // Create input tensor with shape (batch_size, channels, height, width)
        torch::Tensor input = torch::randn({batch_size, channels, height, width});
        
        // Test basic pixel_shuffle operation
        torch::Tensor output = torch::pixel_shuffle(input, upscale_factor);
        
        // Verify output shape is correct
        auto expected_shape = std::vector<int64_t>{
            batch_size, 
            channels / required_factor, 
            height * upscale_factor, 
            width * upscale_factor
        };
        
        if (output.sizes().vec() != expected_shape) {
            std::cerr << "Output shape mismatch!" << std::endl;
        }
        
        // Test with different data types
        if (offset < Size) {
            int dtype_choice = extract_int(Data, Size, offset, 0, 3);
            torch::Tensor typed_input;
            
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
                case 3:
                    typed_input = input.to(torch::kInt64);
                    break;
            }
            
            torch::Tensor typed_output = torch::pixel_shuffle(typed_input, upscale_factor);
        }
        
        // Test edge cases
        // Test with minimum valid upscale factor
        torch::Tensor edge_output1 = torch::pixel_shuffle(input, 1);
        
        // Test with different tensor layouts if we have more data
        if (offset < Size) {
            bool test_contiguous = extract_int(Data, Size, offset, 0, 1) == 1;
            if (!test_contiguous) {
                // Create non-contiguous tensor
                torch::Tensor non_contiguous = input.transpose(2, 3);
                if (non_contiguous.sizes()[1] % required_factor == 0) {
                    torch::Tensor nc_output = torch::pixel_shuffle(non_contiguous, upscale_factor);
                }
            }
        }
        
        // Test with different devices if CUDA is available
        if (torch::cuda::is_available() && offset < Size) {
            bool test_cuda = extract_int(Data, Size, offset, 0, 1) == 1;
            if (test_cuda) {
                torch::Tensor cuda_input = input.to(torch::kCUDA);
                torch::Tensor cuda_output = torch::pixel_shuffle(cuda_input, upscale_factor);
            }
        }
        
        // Test with very small tensors
        if (offset < Size) {
            bool test_small = extract_int(Data, Size, offset, 0, 1) == 1;
            if (test_small) {
                int small_channels = required_factor; // Minimum valid channels
                torch::Tensor small_input = torch::randn({1, small_channels, 1, 1});
                torch::Tensor small_output = torch::pixel_shuffle(small_input, upscale_factor);
            }
        }
        
        // Test with larger upscale factors if we have more data
        if (offset < Size) {
            int large_upscale = extract_int(Data, Size, offset, 2, 6);
            int large_required = large_upscale * large_upscale;
            
            // Create tensor with appropriate channel count
            if (channels >= large_required) {
                int adjusted_channels = (channels / large_required) * large_required;
                torch::Tensor large_input = torch::randn({1, adjusted_channels, 2, 2});
                torch::Tensor large_output = torch::pixel_shuffle(large_input, large_upscale);
            }
        }
        
        // Test gradient computation if we have more data
        if (offset < Size) {
            bool test_grad = extract_int(Data, Size, offset, 0, 1) == 1;
            if (test_grad) {
                torch::Tensor grad_input = input.clone().requires_grad_(true);
                torch::Tensor grad_output = torch::pixel_shuffle(grad_input, upscale_factor);
                torch::Tensor loss = grad_output.sum();
                loss.backward();
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}