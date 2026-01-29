#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cstring>        // For std::memcpy

// --- Fuzzer Entry Point ---
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
        // Need at least a few bytes for meaningful fuzzing
        if (Size < 16) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract padding values from the input data (limit to reasonable values)
        int64_t padding_left = static_cast<int64_t>(Data[offset++]) % 10;
        int64_t padding_right = static_cast<int64_t>(Data[offset++]) % 10;
        int64_t padding_top = static_cast<int64_t>(Data[offset++]) % 10;
        int64_t padding_bottom = static_cast<int64_t>(Data[offset++]) % 10;
        
        // Extract pad value as a simple float to avoid NaN/Inf issues
        float pad_value = static_cast<float>(static_cast<int8_t>(Data[offset++])) / 10.0f;
        
        // Extract dimensions for 4D tensor (N, C, H, W)
        int64_t batch_size = 1 + (Data[offset++] % 4);  // 1-4
        int64_t channels = 1 + (Data[offset++] % 8);    // 1-8
        int64_t height = 1 + (Data[offset++] % 32);     // 1-32
        int64_t width = 1 + (Data[offset++] % 32);      // 1-32
        
        // Determine dtype from fuzzer data
        uint8_t dtype_selector = Data[offset++] % 3;
        torch::ScalarType dtype;
        switch (dtype_selector) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            default: dtype = torch::kFloat32; break;
        }
        
        // Create a properly shaped 4D input tensor for ConstantPad2d
        torch::Tensor input = torch::randn({batch_size, channels, height, width}, 
                                           torch::TensorOptions().dtype(dtype));
        
        // Seed the random generator with fuzzer data for variety
        if (offset + 4 <= Size) {
            uint32_t seed;
            std::memcpy(&seed, Data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            torch::manual_seed(seed);
            input = torch::randn({batch_size, channels, height, width},
                                 torch::TensorOptions().dtype(dtype));
        }
        
        // Create the ConstantPad2d module
        torch::nn::ConstantPad2d pad(
            torch::nn::ConstantPad2dOptions(
                {padding_left, padding_right, padding_top, padding_bottom}, 
                static_cast<double>(pad_value)
            )
        );
        
        // Apply padding to the input tensor
        torch::Tensor output = pad->forward(input);
        
        // Verify output shape is correct
        int64_t expected_height = height + padding_top + padding_bottom;
        int64_t expected_width = width + padding_left + padding_right;
        
        if (output.size(2) != expected_height || output.size(3) != expected_width) {
            std::cerr << "Unexpected output shape" << std::endl;
            return -1;
        }
        
        // Perform operations on the output to ensure it's exercised
        auto sum = output.sum();
        auto mean = output.mean();
        
        // Also test with 3D input (C, H, W)
        if (Data[offset % Size] % 2 == 0) {
            torch::Tensor input_3d = torch::randn({channels, height, width},
                                                   torch::TensorOptions().dtype(dtype));
            torch::Tensor output_3d = pad->forward(input_3d);
            auto sum_3d = output_3d.sum();
            (void)sum_3d;
        }
        
        // Prevent compiler from optimizing away
        (void)sum;
        (void)mean;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}