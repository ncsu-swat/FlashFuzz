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
        
        // Need at least a few bytes for basic parameters
        if (Size < 10) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Need remaining bytes for parameters
        if (offset + 4 >= Size) {
            return 0;
        }
        
        // Extract num_features from the input tensor
        int64_t num_features = 0;
        int input_dim = input.dim();
        
        if (input_dim >= 2) {
            num_features = input.size(1);
        } else if (input_dim == 1) {
            num_features = input.size(0);
        } else {
            num_features = 1;
        }
        
        // Ensure num_features is valid (must be positive)
        if (num_features <= 0) {
            num_features = 1;
        }
        
        // Parse parameters for BatchNorm
        double eps = 1e-5;
        double momentum = 0.1;
        bool affine = true;
        bool track_running_stats = true;
        
        if (offset < Size) {
            // Use a byte to determine eps (small positive value)
            uint8_t eps_byte = Data[offset++];
            eps = static_cast<double>(eps_byte) / 255.0 * 0.1 + 1e-10;
        }
        
        if (offset < Size) {
            // Use a byte to determine momentum (between 0 and 1)
            uint8_t momentum_byte = Data[offset++];
            momentum = static_cast<double>(momentum_byte) / 255.0;
        }
        
        if (offset < Size) {
            // Use a byte to determine boolean parameters
            uint8_t bool_byte = Data[offset++];
            affine = (bool_byte & 0x1);
            track_running_stats = (bool_byte & 0x2);
        }
        
        // Ensure input has float-like dtype for BatchNorm
        if (input.dtype() == torch::kInt8 || 
            input.dtype() == torch::kUInt8 || 
            input.dtype() == torch::kInt16 || 
            input.dtype() == torch::kInt32 || 
            input.dtype() == torch::kInt64 ||
            input.dtype() == torch::kBool) {
            input = input.to(torch::kFloat);
        }
        
        // Handle complex types by taking real part
        if (input.is_complex()) {
            input = torch::real(input);
        }
        
        // Reshape input and select appropriate BatchNorm variant based on dimensions
        // BatchNorm expects input of shape [N, C, ...] where N is batch size and C is num_features
        torch::Tensor output;
        
        if (input_dim <= 2) {
            // Use BatchNorm1d for 1D or 2D input
            if (input_dim == 0) {
                // Scalar tensor, reshape to [1, 1]
                input = input.reshape({1, 1});
                num_features = 1;
            } else if (input_dim == 1) {
                // 1D tensor, reshape to [1, C]
                input = input.reshape({1, input.size(0)});
            }
            
            torch::nn::BatchNorm1d bn1d(
                torch::nn::BatchNorm1dOptions(num_features)
                    .eps(eps)
                    .momentum(momentum)
                    .affine(affine)
                    .track_running_stats(track_running_stats)
            );
            
            // Inner try-catch for expected shape mismatches
            try {
                output = bn1d(input);
                
                // Access properties to ensure the module works
                auto running_mean = bn1d->running_mean;
                auto running_var = bn1d->running_var;
                
                if (affine) {
                    auto weight = bn1d->weight;
                    auto bias = bn1d->bias;
                }
            } catch (...) {
                // Expected failures (shape mismatches, etc.) - silently ignore
            }
        } else if (input_dim == 3) {
            // Use BatchNorm1d for 3D input [N, C, L]
            torch::nn::BatchNorm1d bn1d(
                torch::nn::BatchNorm1dOptions(num_features)
                    .eps(eps)
                    .momentum(momentum)
                    .affine(affine)
                    .track_running_stats(track_running_stats)
            );
            
            try {
                output = bn1d(input);
                
                auto running_mean = bn1d->running_mean;
                auto running_var = bn1d->running_var;
                
                if (affine) {
                    auto weight = bn1d->weight;
                    auto bias = bn1d->bias;
                }
            } catch (...) {
                // Expected failures - silently ignore
            }
        } else if (input_dim == 4) {
            // Use BatchNorm2d for 4D input [N, C, H, W]
            torch::nn::BatchNorm2d bn2d(
                torch::nn::BatchNorm2dOptions(num_features)
                    .eps(eps)
                    .momentum(momentum)
                    .affine(affine)
                    .track_running_stats(track_running_stats)
            );
            
            try {
                output = bn2d(input);
                
                auto running_mean = bn2d->running_mean;
                auto running_var = bn2d->running_var;
                
                if (affine) {
                    auto weight = bn2d->weight;
                    auto bias = bn2d->bias;
                }
            } catch (...) {
                // Expected failures - silently ignore
            }
        } else {
            // Use BatchNorm3d for 5D+ input [N, C, D, H, W]
            // Reshape higher dimensions to 5D
            if (input_dim > 5) {
                auto sizes = input.sizes().vec();
                int64_t batch = sizes[0];
                int64_t channels = sizes[1];
                int64_t remaining = 1;
                for (int i = 2; i < input_dim; i++) {
                    remaining *= sizes[i];
                }
                input = input.reshape({batch, channels, 1, 1, remaining});
            }
            
            torch::nn::BatchNorm3d bn3d(
                torch::nn::BatchNorm3dOptions(num_features)
                    .eps(eps)
                    .momentum(momentum)
                    .affine(affine)
                    .track_running_stats(track_running_stats)
            );
            
            try {
                output = bn3d(input);
                
                auto running_mean = bn3d->running_mean;
                auto running_var = bn3d->running_var;
                
                if (affine) {
                    auto weight = bn3d->weight;
                    auto bias = bn3d->bias;
                }
            } catch (...) {
                // Expected failures - silently ignore
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}