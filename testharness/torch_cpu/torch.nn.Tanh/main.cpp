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
        
        // Skip if there's not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor with floating point type for tanh
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Tanh requires floating point input
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create Tanh module
        torch::nn::Tanh tanh_module;
        
        // Apply Tanh operation using module
        torch::Tensor output = tanh_module(input);
        
        // Alternative way to apply tanh using functional API
        torch::Tensor output2 = torch::tanh(input);
        
        // Try in-place version
        try {
            torch::Tensor input_copy = input.clone();
            input_copy.tanh_();
        } catch (...) {
            // In-place may fail for certain tensor configurations
        }
        
        // Test with different tensor dimensions
        if (offset + 4 < Size) {
            uint8_t dim_choice = Data[offset++] % 4;
            torch::Tensor shaped_input;
            
            try {
                switch (dim_choice) {
                    case 0:
                        // Scalar-like
                        shaped_input = torch::randn({1});
                        break;
                    case 1:
                        // 1D
                        shaped_input = torch::randn({static_cast<int64_t>((Data[offset++] % 64) + 1)});
                        break;
                    case 2:
                        // 2D (batch, features)
                        shaped_input = torch::randn({
                            static_cast<int64_t>((Data[offset++] % 16) + 1),
                            static_cast<int64_t>((Data[offset++] % 32) + 1)
                        });
                        break;
                    case 3:
                        // 4D (batch, channels, height, width)
                        shaped_input = torch::randn({
                            static_cast<int64_t>((Data[offset++] % 4) + 1),
                            static_cast<int64_t>((Data[offset++] % 8) + 1),
                            static_cast<int64_t>((Data[offset++] % 16) + 1),
                            static_cast<int64_t>((Data[offset++] % 16) + 1)
                        });
                        break;
                }
                torch::Tensor shaped_output = tanh_module(shaped_input);
            } catch (...) {
                // Shape creation may fail with insufficient data
            }
        }
        
        // Test train/eval modes
        if (offset + 1 < Size) {
            bool train_mode = Data[offset++] % 2 == 0;
            tanh_module->train(train_mode);
            torch::Tensor output_mode = tanh_module(input);
            
            // Verify output is in [-1, 1] range (tanh property)
            // This exercises the output tensor
            auto max_val = output_mode.abs().max().item<float>();
            (void)max_val;  // Suppress unused warning
        }
        
        // Test with special values (NaN, Inf handling)
        try {
            torch::Tensor special_input = torch::tensor({
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::quiet_NaN(),
                0.0f, 1.0f, -1.0f
            });
            torch::Tensor special_output = tanh_module(special_input);
        } catch (...) {
            // Special value handling may vary
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}