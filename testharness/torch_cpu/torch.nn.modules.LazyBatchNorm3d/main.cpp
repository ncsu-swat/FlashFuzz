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
        
        // Skip if we don't have enough data
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure the input tensor has 5 dimensions (N, C, D, H, W) for BatchNorm3d
        // If not, reshape it to a 5D tensor
        if (input.dim() != 5) {
            // Extract values to determine new shape
            uint8_t num_features = 0;
            if (offset < Size) {
                num_features = Data[offset++] % 64 + 1; // Ensure at least 1 feature
            } else {
                num_features = 3; // Default
            }
            
            // Create a new shape with 5 dimensions
            std::vector<int64_t> new_shape;
            
            // Batch size
            int64_t batch_size = 1;
            if (offset < Size) {
                batch_size = (Data[offset++] % 8) + 1;
            }
            new_shape.push_back(batch_size);
            
            // Channels (num_features)
            new_shape.push_back(num_features);
            
            // Depth, Height, Width
            for (int i = 0; i < 3; i++) {
                int64_t dim_size = 2;
                if (offset < Size) {
                    dim_size = (Data[offset++] % 8) + 1;
                }
                new_shape.push_back(dim_size);
            }
            
            // Reshape or create new tensor
            if (input.numel() > 0) {
                // Calculate total elements in new shape
                int64_t total_elements = 1;
                for (const auto& dim : new_shape) {
                    total_elements *= dim;
                }
                
                // If original tensor has enough elements, reshape it
                if (input.numel() >= total_elements) {
                    input = input.reshape(new_shape);
                } else {
                    // Create new tensor with the desired shape
                    input = torch::ones(new_shape, input.options());
                }
            } else {
                // Create new tensor with the desired shape
                input = torch::ones(new_shape, input.options());
            }
        }
        
        // Get number of features (channels dimension)
        int64_t num_features = input.size(1);
        
        // Create BatchNorm3d module (LazyBatchNorm3d doesn't exist, use regular BatchNorm3d)
        auto bn_options = torch::nn::BatchNorm3dOptions(num_features);
        
        // Configure module parameters based on remaining data
        if (offset + 3 < Size) {
            // Set eps (small value added to variance for numerical stability)
            double eps = static_cast<double>(Data[offset++]) / 255.0 * 0.1;
            bn_options.eps(eps);
            
            // Set momentum
            double momentum = static_cast<double>(Data[offset++]) / 255.0;
            bn_options.momentum(momentum);
            
            // Set affine flag
            bool affine = Data[offset++] % 2 == 0;
            bn_options.affine(affine);
            
            // Set track_running_stats flag
            bool track_running_stats = Data[offset++] % 2 == 0;
            bn_options.track_running_stats(track_running_stats);
        }
        
        torch::nn::BatchNorm3d bn(bn_options);
        
        // Apply the BatchNorm3d to the input tensor
        torch::Tensor output = bn->forward(input);
        
        // Test the module in training and evaluation modes
        bn->train();
        torch::Tensor output_train = bn->forward(input);
        
        bn->eval();
        torch::Tensor output_eval = bn->forward(input);
        
        // Test with different data types if possible
        if (offset < Size && input.scalar_type() != torch::kHalf && input.scalar_type() != torch::kBFloat16) {
            // Try with float16 if available
            try {
                auto input_fp16 = input.to(torch::kHalf);
                auto bn_fp16_options = torch::nn::BatchNorm3dOptions(num_features);
                torch::nn::BatchNorm3d bn_fp16(bn_fp16_options);
                auto output_fp16 = bn_fp16->forward(input_fp16);
            } catch (...) {
                // Ignore errors with half precision
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
