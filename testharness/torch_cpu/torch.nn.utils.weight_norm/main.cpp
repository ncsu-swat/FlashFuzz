#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// Manual implementation of weight normalization computation
// Since torch::nn::utils::weight_norm is not available in C++ frontend,
// we implement the core computation manually
torch::Tensor compute_weight_norm(const torch::Tensor& v, const torch::Tensor& g, int64_t dim) {
    // Compute the norm of v along all dimensions except dim
    std::vector<int64_t> norm_dims;
    for (int64_t i = 0; i < v.dim(); i++) {
        if (i != dim) {
            norm_dims.push_back(i);
        }
    }
    
    // Compute norm and reshape g for broadcasting
    torch::Tensor norm = torch::norm(v, 2, norm_dims, true);
    norm = norm.clamp_min(1e-12); // Avoid division by zero
    
    // Compute normalized weight: w = g * v / ||v||
    return g * v / norm;
}

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
        
        if (Size < 4) {
            return 0;
        }
        
        // Get dimensions from fuzzer data for Linear module
        int64_t in_features = static_cast<int64_t>((Data[offset++] % 15) + 1);  // 1-16
        int64_t out_features = static_cast<int64_t>((Data[offset++] % 15) + 1); // 1-16
        
        // Create a simple linear module
        torch::nn::Linear linear(torch::nn::LinearOptions(in_features, out_features));
        
        // Manually apply weight normalization concept
        // Get the weight parameter
        torch::Tensor weight = linear->weight;
        
        // Compute g (magnitude) and v (direction) for weight normalization
        // Default dim is 0 for weight normalization
        int64_t dim = 0;
        torch::Tensor g = torch::norm(weight, 2, /*dim=*/{1}, /*keepdim=*/true);
        torch::Tensor v = weight.clone();
        
        // Compute normalized weight
        torch::Tensor normalized_weight = compute_weight_norm(v, g, dim);
        
        // Apply the normalized weight (create new tensor, don't modify in-place during grad)
        {
            torch::NoGradGuard no_grad;
            linear->weight.copy_(normalized_weight);
        }
        
        // Verify that forward pass works after manual weight normalization
        auto input = torch::randn({1, in_features});
        auto output = linear->forward(input);
        
        // Test with different dimension values
        if (offset + 1 < Size) {
            int64_t test_dim = static_cast<int64_t>(Data[offset++] % 2);  // Valid dims for 2D weight: 0 or 1
            
            // Recreate linear module for fresh test
            torch::nn::Linear linear2(torch::nn::LinearOptions(in_features, out_features));
            
            try {
                torch::Tensor weight2 = linear2->weight;
                torch::Tensor g2 = torch::norm(weight2, 2, /*dim=*/{1 - test_dim}, /*keepdim=*/true);
                torch::Tensor normalized2 = compute_weight_norm(weight2, g2, test_dim);
                
                {
                    torch::NoGradGuard no_grad;
                    linear2->weight.copy_(normalized2);
                }
                
                auto output2 = linear2->forward(input);
            } catch (...) {
                // May fail for certain dim values, continue
            }
        }
        
        // Create a Conv2d module and test with it
        if (offset + 2 < Size) {
            int64_t in_channels = static_cast<int64_t>((Data[offset++] % 4) + 1);   // 1-4
            int64_t out_channels = static_cast<int64_t>((Data[offset++] % 4) + 1);  // 1-4
            int64_t kernel_size = static_cast<int64_t>((Data[offset++] % 3) + 1);   // 1-3
            
            torch::nn::Conv2d conv(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size));
            
            // Apply manual weight normalization
            torch::Tensor conv_weight = conv->weight;
            // For conv weight [out_channels, in_channels, H, W], normalize along dim 0
            // Flatten all dims except 0 for norm computation
            auto weight_flat = conv_weight.view({conv_weight.size(0), -1});
            torch::Tensor conv_g = torch::norm(weight_flat, 2, /*dim=*/{1}, /*keepdim=*/true);
            conv_g = conv_g.view({conv_weight.size(0), 1, 1, 1});
            torch::Tensor conv_norm = torch::norm(weight_flat, 2, /*dim=*/{1}, /*keepdim=*/true).view({conv_weight.size(0), 1, 1, 1});
            conv_norm = conv_norm.clamp_min(1e-12);
            torch::Tensor conv_normalized = conv_g * conv_weight / conv_norm;
            
            {
                torch::NoGradGuard no_grad;
                conv->weight.copy_(conv_normalized);
            }
            
            // Verify forward pass works
            auto conv_input = torch::randn({1, in_channels, kernel_size + 2, kernel_size + 2});
            auto conv_output = conv->forward(conv_input);
        }
        
        // Create a Conv1d module and test with it
        if (offset + 2 < Size) {
            int64_t in_channels = static_cast<int64_t>((Data[offset++] % 4) + 1);
            int64_t out_channels = static_cast<int64_t>((Data[offset++] % 4) + 1);
            int64_t kernel_size = static_cast<int64_t>((Data[offset++] % 3) + 1);
            
            torch::nn::Conv1d conv1d(torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size));
            
            try {
                // Apply manual weight normalization for Conv1d
                torch::Tensor conv1d_weight = conv1d->weight;
                auto weight1d_flat = conv1d_weight.view({conv1d_weight.size(0), -1});
                torch::Tensor conv1d_g = torch::norm(weight1d_flat, 2, /*dim=*/{1}, /*keepdim=*/true);
                conv1d_g = conv1d_g.view({conv1d_weight.size(0), 1, 1});
                torch::Tensor conv1d_norm = torch::norm(weight1d_flat, 2, /*dim=*/{1}, /*keepdim=*/true).view({conv1d_weight.size(0), 1, 1});
                conv1d_norm = conv1d_norm.clamp_min(1e-12);
                torch::Tensor conv1d_normalized = conv1d_g * conv1d_weight / conv1d_norm;
                
                {
                    torch::NoGradGuard no_grad;
                    conv1d->weight.copy_(conv1d_normalized);
                }
                
                // Verify forward pass works
                auto conv1d_input = torch::randn({1, in_channels, kernel_size + 4});
                auto conv1d_output = conv1d->forward(conv1d_input);
            } catch (...) {
                // Continue on failure
            }
        }
        
        // Create a Conv3d module and test with it
        if (offset + 2 < Size) {
            int64_t in_channels = static_cast<int64_t>((Data[offset++] % 2) + 1);   // 1-2
            int64_t out_channels = static_cast<int64_t>((Data[offset++] % 2) + 1);  // 1-2
            int64_t kernel_size = static_cast<int64_t>((Data[offset++] % 2) + 1);   // 1-2
            
            torch::nn::Conv3d conv3d(torch::nn::Conv3dOptions(in_channels, out_channels, kernel_size));
            
            try {
                // Apply manual weight normalization for Conv3d
                torch::Tensor conv3d_weight = conv3d->weight;
                auto weight3d_flat = conv3d_weight.view({conv3d_weight.size(0), -1});
                torch::Tensor conv3d_g = torch::norm(weight3d_flat, 2, /*dim=*/{1}, /*keepdim=*/true);
                conv3d_g = conv3d_g.view({conv3d_weight.size(0), 1, 1, 1, 1});
                torch::Tensor conv3d_norm = torch::norm(weight3d_flat, 2, /*dim=*/{1}, /*keepdim=*/true).view({conv3d_weight.size(0), 1, 1, 1, 1});
                conv3d_norm = conv3d_norm.clamp_min(1e-12);
                torch::Tensor conv3d_normalized = conv3d_g * conv3d_weight / conv3d_norm;
                
                {
                    torch::NoGradGuard no_grad;
                    conv3d->weight.copy_(conv3d_normalized);
                }
                
                // Verify forward pass works
                auto conv3d_input = torch::randn({1, in_channels, kernel_size + 2, kernel_size + 2, kernel_size + 2});
                auto conv3d_output = conv3d->forward(conv3d_input);
            } catch (...) {
                // Continue on failure
            }
        }
        
        // Test with ConvTranspose2d
        if (offset + 2 < Size) {
            int64_t in_channels = static_cast<int64_t>((Data[offset++] % 4) + 1);
            int64_t out_channels = static_cast<int64_t>((Data[offset++] % 4) + 1);
            int64_t kernel_size = static_cast<int64_t>((Data[offset++] % 3) + 1);
            
            torch::nn::ConvTranspose2d conv_t(torch::nn::ConvTranspose2dOptions(in_channels, out_channels, kernel_size));
            
            try {
                // Apply manual weight normalization for ConvTranspose2d
                // Note: ConvTranspose weight shape is [in_channels, out_channels, H, W]
                torch::Tensor conv_t_weight = conv_t->weight;
                auto weight_t_flat = conv_t_weight.view({conv_t_weight.size(0), -1});
                torch::Tensor conv_t_g = torch::norm(weight_t_flat, 2, /*dim=*/{1}, /*keepdim=*/true);
                conv_t_g = conv_t_g.view({conv_t_weight.size(0), 1, 1, 1});
                torch::Tensor conv_t_norm = torch::norm(weight_t_flat, 2, /*dim=*/{1}, /*keepdim=*/true).view({conv_t_weight.size(0), 1, 1, 1});
                conv_t_norm = conv_t_norm.clamp_min(1e-12);
                torch::Tensor conv_t_normalized = conv_t_g * conv_t_weight / conv_t_norm;
                
                {
                    torch::NoGradGuard no_grad;
                    conv_t->weight.copy_(conv_t_normalized);
                }
                
                auto conv_t_input = torch::randn({1, in_channels, 4, 4});
                auto conv_t_output = conv_t->forward(conv_t_input);
            } catch (...) {
                // Continue on failure
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}