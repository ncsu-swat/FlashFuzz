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
        
        // Early return if not enough data
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input has at least 4 dimensions for InstanceNorm2d (N, C, H, W)
        if (input.dim() < 4) {
            // Reshape to 4D if needed
            std::vector<int64_t> new_shape;
            if (input.dim() == 0) {
                // Scalar to 4D
                new_shape = {1, 1, 1, 1};
            } else if (input.dim() == 1) {
                // 1D to 4D
                new_shape = {1, input.size(0), 1, 1};
            } else if (input.dim() == 2) {
                // 2D to 4D
                new_shape = {1, input.size(0), input.size(1), 1};
            } else if (input.dim() == 3) {
                // 3D to 4D
                new_shape = {1, input.size(0), input.size(1), input.size(2)};
            }
            
            input = input.reshape(new_shape);
        }
        
        // Get number of channels (second dimension)
        int64_t num_channels = input.size(1);
        
        // Ensure we have at least one channel
        if (num_channels < 1) {
            num_channels = 1;
            std::vector<int64_t> new_shape = input.sizes().vec();
            new_shape[1] = 1;
            input = input.reshape(new_shape);
        }
        
        // Convert to quantized tensor if not already
        if (!input.is_quantized()) {
            // Scale and zero point for quantization
            double scale = 0.1;
            int64_t zero_point = 10;
            
            // Quantize the tensor to qint8
            input = torch::quantize_per_tensor(input, scale, zero_point, torch::kQInt8);
        }
        
        // Extract parameters for InstanceNorm2d from the input data
        bool affine = false;
        bool track_running_stats = false;
        double eps = 1e-5;
        double momentum = 0.1;
        
        if (offset + 4 <= Size) {
            // Use 1 byte for each boolean parameter
            affine = Data[offset++] & 0x1;
            track_running_stats = Data[offset++] & 0x1;
            
            // Use 1 byte for eps (scaled)
            if (offset < Size) {
                eps = static_cast<double>(Data[offset++]) / 255.0 * 0.1;
                if (eps < 1e-10) eps = 1e-5; // Avoid too small values
            }
            
            // Use 1 byte for momentum (scaled)
            if (offset < Size) {
                momentum = static_cast<double>(Data[offset++]) / 255.0;
                if (momentum < 0.01) momentum = 0.1; // Avoid too small values
            }
        }
        
        // Create weight and bias if affine is true
        torch::Tensor weight, bias;
        if (affine) {
            // Create weight tensor
            weight = torch::ones({num_channels});
            bias = torch::zeros({num_channels});
            
            // Fill with data from input if available
            if (offset + num_channels * sizeof(float) <= Size) {
                for (int64_t i = 0; i < num_channels && offset + sizeof(float) <= Size; i++) {
                    float val;
                    std::memcpy(&val, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    weight[i] = val;
                }
            }
            
            if (offset + num_channels * sizeof(float) <= Size) {
                for (int64_t i = 0; i < num_channels && offset + sizeof(float) <= Size; i++) {
                    float val;
                    std::memcpy(&val, Data + offset, sizeof(float));
                    offset += sizeof(float);
                    bias[i] = val;
                }
            }
        }
        
        // Create InstanceNorm2d module
        torch::nn::InstanceNorm2dOptions options(num_channels);
        options.eps(eps);
        options.momentum(momentum);
        options.affine(affine);
        options.track_running_stats(track_running_stats);
        
        auto instance_norm = torch::nn::InstanceNorm2d(options);
        
        // Set weight and bias if affine is true
        if (affine) {
            instance_norm->weight = weight;
            instance_norm->bias = bias;
        }
        
        // Apply InstanceNorm2d to the input tensor
        torch::Tensor output = instance_norm->forward(input);
        
        // Perform some operations on the output to ensure it's used
        auto output_sum = output.sum();
        
        // Dequantize for further operations if needed
        auto dequantized = output.dequantize();
        
        // Test with different batch sizes
        if (offset + 1 < Size && input.size(0) > 1) {
            // Try with a single sample
            auto single_input = input.slice(0, 0, 1);
            auto single_output = instance_norm->forward(single_input);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
