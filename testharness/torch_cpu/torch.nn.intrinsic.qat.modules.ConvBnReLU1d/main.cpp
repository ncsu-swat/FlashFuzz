#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for ConvBnReLU1d from the remaining data
        if (offset + 8 >= Size) {
            return 0;
        }
        
        // Parse in_channels, out_channels, kernel_size
        int64_t in_channels = 1 + (Data[offset++] % 16);
        int64_t out_channels = 1 + (Data[offset++] % 16);
        int64_t kernel_size = 1 + (Data[offset++] % 7);
        
        // Parse stride, padding, dilation, groups
        int64_t stride = 1 + (Data[offset++] % 3);
        int64_t padding = Data[offset++] % 4;
        int64_t dilation = 1 + (Data[offset++] % 2);
        int64_t groups = 1;
        
        // If there's enough data, parse groups (ensuring it's a divisor of in_channels)
        if (offset < Size) {
            groups = 1 + (Data[offset++] % std::max(1, static_cast<int>(in_channels)));
            if (in_channels % groups != 0) {
                groups = 1;
            }
        }
        
        // Parse QAT parameters
        double scale = 1.0;
        int64_t zero_point = 0;
        
        if (offset + 8 < Size) {
            // Extract scale and zero_point from remaining data
            memcpy(&scale, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Ensure scale is positive and not too extreme
            scale = std::abs(scale);
            if (scale < 1e-10) scale = 1e-10;
            if (scale > 1e10) scale = 1e10;
            
            memcpy(&zero_point, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure zero_point is in a reasonable range
            zero_point = zero_point % 256;
        }
        
        // Reshape input tensor if needed to match expected input shape for Conv1d
        // Conv1d expects input of shape [batch_size, in_channels, sequence_length]
        if (input.dim() < 3) {
            // If tensor has fewer than 3 dimensions, reshape it
            std::vector<int64_t> new_shape;
            
            if (input.dim() == 0) {
                // Scalar tensor, reshape to [1, in_channels, 1]
                new_shape = {1, in_channels, 1};
            } else if (input.dim() == 1) {
                // 1D tensor, reshape to [1, in_channels, length]
                int64_t length = input.size(0) / in_channels;
                if (length < 1) length = 1;
                new_shape = {1, in_channels, length};
            } else if (input.dim() == 2) {
                // 2D tensor, reshape to [batch, in_channels, length]
                int64_t batch = input.size(0);
                int64_t length = input.size(1) / in_channels;
                if (length < 1) length = 1;
                new_shape = {batch, in_channels, length};
            }
            
            // Resize the tensor to the new shape
            try {
                input = input.reshape(new_shape);
            } catch (const std::exception&) {
                // If reshape fails, create a new tensor with the desired shape
                input = torch::ones(new_shape, input.options());
            }
        } else if (input.size(1) != in_channels) {
            // If the tensor has the right number of dimensions but wrong channel count
            std::vector<int64_t> new_shape = {input.size(0), in_channels, input.size(2)};
            try {
                input = input.reshape(new_shape);
            } catch (const std::exception&) {
                input = torch::ones(new_shape, input.options());
            }
        }
        
        // Create a quantized tensor for QAT
        torch::Tensor q_input;
        try {
            q_input = torch::quantize_per_tensor(input.to(torch::kFloat), scale, zero_point, torch::kQUInt8);
        } catch (const std::exception&) {
            // If quantization fails, use a default quantized tensor
            q_input = torch::quantize_per_tensor(torch::ones_like(input.to(torch::kFloat)), 1.0, 0, torch::kQUInt8);
        }
        
        // Create ConvBnReLU1d module options
        auto conv_options = torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                                            .stride(stride)
                                            .padding(padding)
                                            .dilation(dilation)
                                            .groups(groups)
                                            .bias(true);
        
        // Create BatchNorm1d options
        auto bn_options = torch::nn::BatchNorm1dOptions(out_channels);
        
        // Create individual modules for testing (since ConvBnReLU1d QAT may not be available)
        torch::nn::Conv1d conv(conv_options);
        torch::nn::BatchNorm1d bn(bn_options);
        
        // Set to training mode for QAT
        conv->train();
        bn->train();
        
        // Forward pass through conv -> bn -> relu
        torch::Tensor output;
        try {
            torch::Tensor conv_out = conv->forward(input);
            torch::Tensor bn_out = bn->forward(conv_out);
            output = torch::relu(bn_out);
        } catch (const std::exception&) {
            // If forward fails, try with a standard-sized input
            torch::Tensor fallback_input = torch::ones({1, in_channels, 10}, torch::kFloat);
            torch::Tensor conv_out = conv->forward(fallback_input);
            torch::Tensor bn_out = bn->forward(conv_out);
            output = torch::relu(bn_out);
        }
        
        // Try to set to eval mode (simulating freeze)
        try {
            conv->eval();
            bn->eval();
        } catch (const std::exception&) {
            // Setting eval mode might fail, that's okay for testing
        }
        
        // Try another forward pass after setting to eval mode
        try {
            torch::Tensor conv_out = conv->forward(input);
            torch::Tensor bn_out = bn->forward(conv_out);
            torch::Tensor frozen_output = torch::relu(bn_out);
        } catch (const std::exception&) {
            // Forward after eval mode might fail, that's okay
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}