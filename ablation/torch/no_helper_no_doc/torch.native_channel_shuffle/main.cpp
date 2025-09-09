#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least enough data for tensor dimensions and groups parameter
        if (Size < 16) {
            return 0;
        }

        // Extract tensor dimensions (4D tensor for channel shuffle)
        int64_t batch_size = extract_int64_t(Data, Size, offset) % 10 + 1;  // 1-10
        int64_t channels = extract_int64_t(Data, Size, offset) % 64 + 1;    // 1-64
        int64_t height = extract_int64_t(Data, Size, offset) % 32 + 1;      // 1-32
        int64_t width = extract_int64_t(Data, Size, offset) % 32 + 1;       // 1-32

        // Extract groups parameter - must be a divisor of channels
        int64_t groups = extract_int64_t(Data, Size, offset) % channels + 1;
        
        // Ensure groups divides channels evenly
        while (channels % groups != 0 && groups > 1) {
            groups--;
        }
        if (groups == 0) groups = 1;

        // Extract data type
        int dtype_choice = extract_int64_t(Data, Size, offset) % 4;
        torch::ScalarType dtype;
        switch (dtype_choice) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kInt32; break;
            default: dtype = torch::kInt64; break;
        }

        // Create input tensor with the specified dimensions
        torch::Tensor input = torch::randn({batch_size, channels, height, width}, 
                                         torch::TensorOptions().dtype(dtype));

        // Test native_channel_shuffle with valid parameters
        torch::Tensor result = torch::native_channel_shuffle(input, groups);

        // Verify output shape is correct
        auto input_sizes = input.sizes();
        auto result_sizes = result.sizes();
        
        if (input_sizes.size() != result_sizes.size()) {
            std::cerr << "Shape mismatch: input and output have different number of dimensions" << std::endl;
            return -1;
        }

        for (size_t i = 0; i < input_sizes.size(); ++i) {
            if (input_sizes[i] != result_sizes[i]) {
                std::cerr << "Shape mismatch at dimension " << i << std::endl;
                return -1;
            }
        }

        // Test edge cases if we have enough data
        if (offset < Size - 8) {
            // Test with groups = 1 (should be identity-like operation)
            torch::Tensor result_groups_1 = torch::native_channel_shuffle(input, 1);
            
            // Test with groups = channels (each channel becomes its own group)
            if (channels > 1) {
                torch::Tensor result_groups_channels = torch::native_channel_shuffle(input, channels);
            }

            // Test with different tensor layouts if possible
            if (input.is_contiguous()) {
                torch::Tensor non_contiguous = input.transpose(2, 3);
                if (!non_contiguous.is_contiguous()) {
                    torch::Tensor result_non_contiguous = torch::native_channel_shuffle(non_contiguous, groups);
                }
            }
        }

        // Test with different input shapes if we have more data
        if (offset < Size - 16) {
            // Test 3D tensor (no batch dimension)
            torch::Tensor input_3d = torch::randn({channels, height, width}, 
                                                torch::TensorOptions().dtype(dtype));
            torch::Tensor result_3d = torch::native_channel_shuffle(input_3d, groups);

            // Test 5D tensor
            int64_t depth = extract_int64_t(Data, Size, offset) % 8 + 1;
            torch::Tensor input_5d = torch::randn({batch_size, channels, depth, height, width}, 
                                                torch::TensorOptions().dtype(dtype));
            torch::Tensor result_5d = torch::native_channel_shuffle(input_5d, groups);
        }

        // Test error conditions that should throw exceptions
        if (offset < Size - 8) {
            try {
                // Test with groups = 0 (should fail)
                torch::native_channel_shuffle(input, 0);
            } catch (const std::exception&) {
                // Expected to throw
            }

            try {
                // Test with groups that don't divide channels evenly
                if (channels > 1) {
                    int64_t bad_groups = channels + 1;
                    while (channels % bad_groups == 0) {
                        bad_groups++;
                    }
                    torch::native_channel_shuffle(input, bad_groups);
                }
            } catch (const std::exception&) {
                // Expected to throw
            }

            try {
                // Test with negative groups
                torch::native_channel_shuffle(input, -1);
            } catch (const std::exception&) {
                // Expected to throw
            }
        }

        // Force computation to ensure no lazy evaluation issues
        result.sum().item<double>();

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}