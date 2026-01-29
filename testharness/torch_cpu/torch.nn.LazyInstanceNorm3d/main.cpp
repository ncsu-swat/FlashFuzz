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
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor for InstanceNorm3d
        // This should be a 5D tensor (N, C, D, H, W)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for InstanceNorm3d from the remaining data
        bool affine = false;
        bool track_running_stats = false;
        double eps = 1e-5;
        double momentum = 0.1;
        int64_t num_features = 1;
        
        if (offset + 2 <= Size) {
            affine = Data[offset++] & 0x1;
            track_running_stats = Data[offset++] & 0x1;
        }
        
        // Extract num_features (channels) from fuzzer data
        if (offset + 1 <= Size) {
            num_features = (Data[offset++] % 16) + 1;  // 1 to 16 channels
        }
        
        if (offset + sizeof(double) <= Size) {
            memcpy(&eps, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure eps is positive and not too small
            eps = std::abs(eps);
            if (eps < 1e-10) eps = 1e-5;
            if (std::isnan(eps) || std::isinf(eps)) eps = 1e-5;
        }
        
        if (offset + sizeof(double) <= Size) {
            memcpy(&momentum, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Ensure momentum is in [0, 1]
            momentum = std::abs(momentum);
            if (std::isnan(momentum) || std::isinf(momentum)) momentum = 0.1;
            if (momentum > 1.0) momentum = momentum - std::floor(momentum);
        }
        
        // Ensure input is 5D for InstanceNorm3d (N, C, D, H, W)
        // Need at least 2 elements in each spatial dimension for normalization
        if (input.dim() < 5) {
            std::vector<int64_t> new_shape;
            for (int i = 0; i < 5; ++i) {
                if (i < input.dim()) {
                    new_shape.push_back(std::max(input.size(i), (int64_t)1));
                } else {
                    new_shape.push_back(1);
                }
            }
            input = input.reshape(new_shape);
        } else if (input.dim() > 5) {
            // Flatten extra dimensions into the batch dimension
            int64_t batch_size = 1;
            for (int i = 0; i < input.dim() - 4; ++i) {
                batch_size *= input.size(i);
            }
            input = input.reshape({batch_size, input.size(-4), input.size(-3), input.size(-2), input.size(-1)});
        }
        
        // Ensure we have valid dimensions (at least 1 in each dimension)
        if (input.numel() == 0) {
            return 0;
        }
        
        // Reshape input to have exactly num_features channels
        // Calculate total elements and redistribute
        int64_t total_elements = input.numel();
        int64_t batch_size = input.size(0);
        int64_t spatial_elements = total_elements / (batch_size * num_features);
        
        if (spatial_elements < 1) {
            // Not enough elements, create a minimal valid tensor
            input = torch::randn({1, num_features, 2, 2, 2});
        } else {
            // Try to reshape to valid dimensions
            // Find factors for D, H, W
            int64_t d = 1, h = 1, w = spatial_elements;
            for (int64_t i = 2; i * i <= spatial_elements; ++i) {
                if (spatial_elements % i == 0) {
                    d = i;
                    int64_t remaining = spatial_elements / i;
                    for (int64_t j = 2; j * j <= remaining; ++j) {
                        if (remaining % j == 0) {
                            h = j;
                            w = remaining / j;
                            break;
                        }
                    }
                    if (h > 1) break;
                }
            }
            
            try {
                input = input.reshape({batch_size, num_features, d, h, w});
            } catch (...) {
                // If reshape fails, create a valid tensor
                input = torch::randn({1, num_features, 2, 2, 2});
            }
        }
        
        // Convert to float for normalization
        input = input.to(torch::kFloat32);
        
        // Create InstanceNorm3d module with explicit num_features
        // Note: LazyInstanceNorm3d doesn't exist in C++ frontend, use InstanceNorm3d instead
        torch::nn::InstanceNorm3d norm(
            torch::nn::InstanceNorm3dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Apply the operation
        torch::Tensor output = norm(input);
        
        // Force evaluation
        output = output.clone();
        
        // Access some properties to ensure computation
        auto sizes = output.sizes();
        (void)sizes;
        
        // Try to access the first element if tensor is not empty
        if (output.numel() > 0) {
            volatile float first_elem = output.flatten()[0].item<float>();
            (void)first_elem;
        }
        
        // Test with a second input to exercise the module further
        if (offset + 4 <= Size) {
            torch::Tensor input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Reshape to 5D with same number of channels
            if (input2.numel() >= num_features) {
                try {
                    int64_t total_elements2 = input2.numel();
                    int64_t elements_per_channel = total_elements2 / num_features;
                    if (elements_per_channel >= 1) {
                        // Try to reshape maintaining channel count
                        input2 = input2.reshape({1, num_features, elements_per_channel, 1, 1}).to(torch::kFloat32);
                        torch::Tensor output2 = norm(input2);
                        output2 = output2.clone();
                    }
                } catch (...) {
                    // Shape mismatch is expected for some inputs
                }
            }
        }
        
        // Test training vs eval mode
        norm->train();
        try {
            torch::Tensor output_train = norm(input);
            output_train = output_train.clone();
        } catch (...) {
            // May fail depending on track_running_stats
        }
        
        norm->eval();
        try {
            torch::Tensor output_eval = norm(input);
            output_eval = output_eval.clone();
        } catch (...) {
            // May fail depending on track_running_stats
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}