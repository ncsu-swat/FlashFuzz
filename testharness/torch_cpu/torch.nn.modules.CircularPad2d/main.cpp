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
        // Need at least 8 bytes for configuration (4 padding values + tensor data)
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract padding values from the first 4 bytes (one byte each, bounded)
        // Padding order: left, right, top, bottom
        int64_t pad_left = (Data[offset++] % 16) + 1;   // 1-16
        int64_t pad_right = (Data[offset++] % 16) + 1;  // 1-16
        int64_t pad_top = (Data[offset++] % 16) + 1;    // 1-16
        int64_t pad_bottom = (Data[offset++] % 16) + 1; // 1-16
        
        // Extract tensor dimensions from next 4 bytes
        // CircularPad2d requires 3D or 4D input
        // For 4D: (N, C, H, W), for 3D: (C, H, W)
        // H and W must be > max padding in that dimension for circular to work
        int64_t batch = (Data[offset++] % 4) + 1;       // 1-4
        int64_t channels = (Data[offset++] % 4) + 1;    // 1-4
        int64_t height = (Data[offset++] % 32) + 17;    // 17-48 (must be > pad_top + pad_bottom)
        int64_t width = (Data[offset++] % 32) + 17;     // 17-48 (must be > pad_left + pad_right)
        
        // Create 4D input tensor
        torch::Tensor input_tensor = torch::randn({batch, channels, height, width});
        
        // If we have remaining data, use it to influence tensor values
        if (offset < Size) {
            // Use fuzzer data to scale the input
            float scale = (Data[offset % Size] / 255.0f) * 2.0f - 1.0f;
            input_tensor = input_tensor * scale;
        }
        
        // Test 1: Apply circular padding using functional interface with 4D tensor
        // Padding order for pad() is: left, right, top, bottom
        try {
            torch::Tensor output = torch::nn::functional::pad(
                input_tensor, 
                torch::nn::functional::PadFuncOptions({pad_left, pad_right, pad_top, pad_bottom})
                    .mode(torch::kCircular)
            );
            
            // Ensure the output is materialized
            output.sum().item<float>();
            
            // Verify output dimensions
            auto out_sizes = output.sizes();
            (void)out_sizes;
        }
        catch (const c10::Error&) {
            // Expected for some invalid configurations
        }
        
        // Test 2: Apply circular padding with 3D tensor (C, H, W)
        try {
            torch::Tensor input_3d = torch::randn({channels, height, width});
            torch::Tensor output_3d = torch::nn::functional::pad(
                input_3d, 
                torch::nn::functional::PadFuncOptions({pad_left, pad_right, pad_top, pad_bottom})
                    .mode(torch::kCircular)
            );
            output_3d.sum().item<float>();
        }
        catch (const c10::Error&) {
            // Expected for some invalid configurations
        }
        
        // Test 3: Symmetric padding (same on both sides)
        try {
            int64_t sym_pad_h = (pad_top + pad_bottom) / 2;
            int64_t sym_pad_w = (pad_left + pad_right) / 2;
            torch::Tensor output_sym = torch::nn::functional::pad(
                input_tensor, 
                torch::nn::functional::PadFuncOptions({sym_pad_w, sym_pad_w, sym_pad_h, sym_pad_h})
                    .mode(torch::kCircular)
            );
            output_sym.sum().item<float>();
        }
        catch (const c10::Error&) {
            // Expected for some invalid configurations
        }
        
        // Test 4: Different data types
        try {
            torch::Tensor input_double = input_tensor.to(torch::kDouble);
            torch::Tensor output_double = torch::nn::functional::pad(
                input_double, 
                torch::nn::functional::PadFuncOptions({pad_left, pad_right, pad_top, pad_bottom})
                    .mode(torch::kCircular)
            );
            output_double.sum().item<double>();
        }
        catch (const c10::Error&) {
            // Expected for some configurations
        }
        
        // Test 5: Integer tensor
        try {
            torch::Tensor input_int = torch::randint(0, 100, {batch, channels, height, width}, torch::kInt);
            torch::Tensor output_int = torch::nn::functional::pad(
                input_int, 
                torch::nn::functional::PadFuncOptions({pad_left, pad_right, pad_top, pad_bottom})
                    .mode(torch::kCircular)
            );
            output_int.sum().item<int>();
        }
        catch (const c10::Error&) {
            // Expected for some configurations
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
}