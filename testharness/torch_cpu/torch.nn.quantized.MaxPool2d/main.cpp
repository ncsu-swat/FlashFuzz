#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

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
        size_t offset = 0;
        
        if (Size < 16) {
            return 0;
        }
        
        // Parse quantization parameters first
        float scale = 0.1f;
        int zero_point = 10;
        
        if (offset + sizeof(float) + sizeof(int) <= Size) {
            memcpy(&scale, Data + offset, sizeof(float));
            offset += sizeof(float);
            memcpy(&zero_point, Data + offset, sizeof(int));
            offset += sizeof(int);
        }
        
        // Ensure scale is positive and reasonable
        scale = std::abs(scale);
        if (!std::isfinite(scale) || scale < 1e-6f) scale = 1e-6f;
        if (scale > 1e6f) scale = 1e6f;
        
        // Ensure zero_point is within valid range for quint8
        zero_point = std::abs(zero_point) % 256;
        
        // Parse MaxPool2d parameters from the input data
        int kernel_size = 2;
        int stride = 1;
        int padding = 0;
        int dilation = 1;
        bool ceil_mode = false;
        
        if (offset + 5 <= Size) {
            kernel_size = (Data[offset++] % 5) + 1;  // 1 to 5
            stride = (Data[offset++] % 3) + 1;       // 1 to 3
            padding = Data[offset++] % 3;            // 0 to 2
            dilation = (Data[offset++] % 2) + 1;     // 1 to 2
            ceil_mode = (Data[offset++] % 2 == 1);
        }
        
        // Parse dimensions for 4D input (N, C, H, W) - required for MaxPool2d
        int batch = 1;
        int channels = 1;
        int height = 8;
        int width = 8;
        
        if (offset + 4 <= Size) {
            batch = (Data[offset++] % 4) + 1;      // 1 to 4
            channels = (Data[offset++] % 4) + 1;   // 1 to 4
            height = (Data[offset++] % 16) + kernel_size + 2;  // Ensure valid size
            width = (Data[offset++] % 16) + kernel_size + 2;   // Ensure valid size
        }
        
        // Ensure dimensions are valid for the pooling operation
        int effective_kernel = kernel_size + (kernel_size - 1) * (dilation - 1);
        if (height < effective_kernel) height = effective_kernel + 1;
        if (width < effective_kernel) width = effective_kernel + 1;
        
        // Create input tensor with proper 4D shape for MaxPool2d
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor input_tensor;
        
        // Use remaining data to fill tensor if available
        if (offset < Size) {
            size_t remaining = Size - offset;
            size_t num_elements = batch * channels * height * width;
            std::vector<float> data_vec(num_elements);
            
            for (size_t i = 0; i < num_elements; i++) {
                // Use fuzzer data to generate values
                uint8_t byte_val = Data[offset + (i % remaining)];
                data_vec[i] = static_cast<float>(byte_val) / 255.0f * 2.0f - 1.0f;
            }
            
            input_tensor = torch::from_blob(data_vec.data(), {batch, channels, height, width}, options).clone();
        } else {
            input_tensor = torch::rand({batch, channels, height, width}, options);
        }
        
        // Quantize the tensor
        torch::Tensor q_input;
        try {
            q_input = torch::quantize_per_tensor(input_tensor, scale, zero_point, torch::kQUInt8);
        } catch (...) {
            // If quantization fails with given params, use safe defaults
            q_input = torch::quantize_per_tensor(input_tensor, 0.1, 10, torch::kQUInt8);
        }
        
        // Apply max_pool2d on quantized tensor
        // This is the C++ equivalent of torch.nn.quantized.MaxPool2d
        torch::Tensor output;
        try {
            output = torch::max_pool2d(q_input, 
                                       {kernel_size, kernel_size}, 
                                       {stride, stride}, 
                                       {padding, padding}, 
                                       {dilation, dilation}, 
                                       ceil_mode);
        } catch (...) {
            // Shape mismatch or invalid params - silently ignore
            return 0;
        }
        
        // Verify output is still quantized and valid
        (void)output.sizes();
        (void)output.is_quantized();
        
        // Dequantize to verify the operation completed correctly
        torch::Tensor dequantized = output.dequantize();
        (void)dequantized.sizes();
        
        // Test different kernel shapes (non-square)
        if (offset + 2 <= Size) {
            int kh = (Data[offset] % 3) + 1;
            int kw = (Data[offset + 1] % 3) + 1;
            
            try {
                torch::Tensor output2 = torch::max_pool2d(q_input,
                                                          {kh, kw},
                                                          {1, 1},
                                                          {0, 0},
                                                          {1, 1},
                                                          false);
                (void)output2.sizes();
            } catch (...) {
                // Silently ignore invalid configurations
            }
        }
        
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
}