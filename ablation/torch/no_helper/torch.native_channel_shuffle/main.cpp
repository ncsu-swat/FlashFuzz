#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least enough data for tensor dimensions and groups
        if (Size < 20) return 0;

        // Extract tensor dimensions (batch, channels, height, width)
        int batch = extractInt(Data, Size, offset, 1, 8);
        int channels = extractInt(Data, Size, offset, 1, 64);
        int height = extractInt(Data, Size, offset, 1, 32);
        int width = extractInt(Data, Size, offset, 1, 32);
        
        // Extract groups parameter
        int groups = extractInt(Data, Size, offset, 1, channels);
        
        // Ensure channels is divisible by groups for valid channel shuffle
        if (channels % groups != 0) {
            // Adjust channels to be divisible by groups
            channels = (channels / groups) * groups;
            if (channels == 0) channels = groups;
        }

        // Create input tensor with random data
        torch::Tensor input = torch::randn({batch, channels, height, width});
        
        // Test different data types
        std::vector<torch::ScalarType> dtypes = {
            torch::kFloat32, torch::kFloat64, torch::kFloat16,
            torch::kInt32, torch::kInt64, torch::kInt8, torch::kUInt8
        };
        
        torch::ScalarType dtype = dtypes[extractInt(Data, Size, offset, 0, dtypes.size() - 1)];
        input = input.to(dtype);

        // Test with different devices if CUDA is available
        std::vector<torch::Device> devices = {torch::kCPU};
        if (torch::cuda::is_available()) {
            devices.push_back(torch::kCUDA);
        }
        
        torch::Device device = devices[extractInt(Data, Size, offset, 0, devices.size() - 1)];
        input = input.to(device);

        // Call native_channel_shuffle
        torch::Tensor output = torch::native_channel_shuffle(input, groups);
        
        // Verify output shape matches input shape
        if (!output.sizes().equals(input.sizes())) {
            std::cerr << "Output shape mismatch!" << std::endl;
        }
        
        // Verify output device and dtype match input
        if (output.device() != input.device() || output.dtype() != input.dtype()) {
            std::cerr << "Output device or dtype mismatch!" << std::endl;
        }

        // Test edge cases
        if (offset < Size) {
            // Test with groups = 1 (should be identity operation)
            torch::Tensor output_identity = torch::native_channel_shuffle(input, 1);
            if (!torch::allclose(input, output_identity, 1e-5, 1e-8, /*equal_nan=*/true)) {
                std::cerr << "Identity test failed!" << std::endl;
            }
            
            // Test with groups = channels (each channel becomes its own group)
            if (channels > 1) {
                torch::Tensor output_max_groups = torch::native_channel_shuffle(input, channels);
                if (!output_max_groups.sizes().equals(input.sizes())) {
                    std::cerr << "Max groups test failed!" << std::endl;
                }
            }
        }

        // Test with different memory layouts
        if (offset < Size) {
            bool test_contiguous = extractBool(Data, Size, offset);
            if (!test_contiguous) {
                // Create non-contiguous tensor
                torch::Tensor non_contiguous = input.transpose(1, 2);
                if (!non_contiguous.is_contiguous()) {
                    torch::Tensor output_nc = torch::native_channel_shuffle(non_contiguous, groups);
                    // Should still work with non-contiguous input
                }
            }
        }

        // Test with zero-sized dimensions
        if (offset < Size) {
            bool test_zero_size = extractBool(Data, Size, offset);
            if (test_zero_size && height > 1 && width > 1) {
                torch::Tensor zero_height = torch::randn({batch, channels, 0, width}).to(dtype).to(device);
                torch::Tensor output_zero = torch::native_channel_shuffle(zero_height, groups);
                if (output_zero.size(2) != 0) {
                    std::cerr << "Zero height test failed!" << std::endl;
                }
            }
        }

        // Test gradient computation if input requires grad
        if (offset < Size && input.dtype().isFloatingType()) {
            bool test_grad = extractBool(Data, Size, offset);
            if (test_grad) {
                torch::Tensor grad_input = input.clone().requires_grad_(true);
                torch::Tensor grad_output = torch::native_channel_shuffle(grad_input, groups);
                
                // Compute some loss and backward
                torch::Tensor loss = grad_output.sum();
                loss.backward();
                
                // Check that gradients exist
                if (!grad_input.grad().defined()) {
                    std::cerr << "Gradient computation failed!" << std::endl;
                }
            }
        }

        // Force evaluation to catch any lazy execution issues
        output.sum().item<double>();

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}