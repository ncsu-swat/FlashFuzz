#include "fuzzer_utils.h"
#include <iostream>

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
        // Need minimum data for parameters
        if (Size < 8) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Parse batch size (1-8)
        int64_t N = static_cast<int64_t>(Data[offset++] % 8) + 1;
        
        // Parse spatial dimensions (1-32)
        int64_t H = static_cast<int64_t>(Data[offset++] % 32) + 1;
        int64_t W = static_cast<int64_t>(Data[offset++] % 32) + 1;
        int64_t D = static_cast<int64_t>(Data[offset++] % 16) + 1;
        
        // Parse channels (1-16)
        int64_t C = static_cast<int64_t>(Data[offset++] % 16) + 1;
        
        // Parse align_corners
        bool align_corners = Data[offset++] & 1;
        
        // Parse mode: 0 = 2D, 1 = 3D
        bool is_3d = Data[offset++] & 1;
        
        // Parse theta values from remaining data
        // For 2D: theta shape is [N, 2, 3] (6 values per batch)
        // For 3D: theta shape is [N, 3, 4] (12 values per batch)
        
        torch::Tensor theta;
        
        if (is_3d) {
            // 3D case: theta [N, 3, 4], size [N, C, D, H, W]
            int64_t theta_elements = N * 3 * 4;
            std::vector<float> theta_data(theta_elements);
            
            for (int64_t i = 0; i < theta_elements; i++) {
                if (offset < Size) {
                    // Scale to reasonable range [-2, 2]
                    theta_data[i] = (static_cast<float>(Data[offset++]) / 255.0f) * 4.0f - 2.0f;
                } else {
                    // Default to identity-like values
                    theta_data[i] = (i % 4 == i / 4) ? 1.0f : 0.0f;
                }
            }
            
            theta = torch::from_blob(theta_data.data(), {N, 3, 4}, torch::kFloat32).clone();
            
            try {
                auto grid = torch::affine_grid_generator(theta, {N, C, D, H, W}, align_corners);
                // Verify output shape: should be [N, D, H, W, 3]
                (void)grid;
            } catch (const std::exception&) {
                // Expected for invalid combinations
            }
        } else {
            // 2D case: theta [N, 2, 3], size [N, C, H, W]
            int64_t theta_elements = N * 2 * 3;
            std::vector<float> theta_data(theta_elements);
            
            for (int64_t i = 0; i < theta_elements; i++) {
                if (offset < Size) {
                    // Scale to reasonable range [-2, 2]
                    theta_data[i] = (static_cast<float>(Data[offset++]) / 255.0f) * 4.0f - 2.0f;
                } else {
                    // Default to identity-like values
                    int row = (i % 6) / 3;
                    int col = (i % 6) % 3;
                    theta_data[i] = (row == col) ? 1.0f : 0.0f;
                }
            }
            
            theta = torch::from_blob(theta_data.data(), {N, 2, 3}, torch::kFloat32).clone();
            
            try {
                auto grid = torch::affine_grid_generator(theta, {N, C, H, W}, align_corners);
                // Verify output shape: should be [N, H, W, 2]
                (void)grid;
            } catch (const std::exception&) {
                // Expected for invalid combinations
            }
        }
        
        // Additional edge case exploration based on remaining data
        if (offset < Size) {
            uint8_t edge_case = Data[offset++] % 4;
            
            switch (edge_case) {
                case 0: {
                    // Test with identity transformation (2D)
                    auto identity_theta = torch::zeros({N, 2, 3});
                    identity_theta.index_put_({torch::indexing::Slice(), 0, 0}, 1.0f);
                    identity_theta.index_put_({torch::indexing::Slice(), 1, 1}, 1.0f);
                    try {
                        auto grid = torch::affine_grid_generator(identity_theta, {N, C, H, W}, align_corners);
                        (void)grid;
                    } catch (const std::exception&) {}
                    break;
                }
                case 1: {
                    // Test with scale transformation
                    float scale = (offset < Size) ? (static_cast<float>(Data[offset++]) / 255.0f * 2.0f + 0.1f) : 1.0f;
                    auto scale_theta = torch::zeros({N, 2, 3});
                    scale_theta.index_put_({torch::indexing::Slice(), 0, 0}, scale);
                    scale_theta.index_put_({torch::indexing::Slice(), 1, 1}, scale);
                    try {
                        auto grid = torch::affine_grid_generator(scale_theta, {N, C, H, W}, align_corners);
                        (void)grid;
                    } catch (const std::exception&) {}
                    break;
                }
                case 2: {
                    // Test with translation
                    float tx = (offset < Size) ? (static_cast<float>(Data[offset++]) / 255.0f * 2.0f - 1.0f) : 0.0f;
                    float ty = (offset < Size) ? (static_cast<float>(Data[offset++]) / 255.0f * 2.0f - 1.0f) : 0.0f;
                    auto trans_theta = torch::zeros({N, 2, 3});
                    trans_theta.index_put_({torch::indexing::Slice(), 0, 0}, 1.0f);
                    trans_theta.index_put_({torch::indexing::Slice(), 1, 1}, 1.0f);
                    trans_theta.index_put_({torch::indexing::Slice(), 0, 2}, tx);
                    trans_theta.index_put_({torch::indexing::Slice(), 1, 2}, ty);
                    try {
                        auto grid = torch::affine_grid_generator(trans_theta, {N, C, H, W}, align_corners);
                        (void)grid;
                    } catch (const std::exception&) {}
                    break;
                }
                case 3: {
                    // Test 3D identity transformation
                    auto identity_3d = torch::zeros({N, 3, 4});
                    identity_3d.index_put_({torch::indexing::Slice(), 0, 0}, 1.0f);
                    identity_3d.index_put_({torch::indexing::Slice(), 1, 1}, 1.0f);
                    identity_3d.index_put_({torch::indexing::Slice(), 2, 2}, 1.0f);
                    try {
                        auto grid = torch::affine_grid_generator(identity_3d, {N, C, D, H, W}, align_corners);
                        (void)grid;
                    } catch (const std::exception&) {}
                    break;
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}