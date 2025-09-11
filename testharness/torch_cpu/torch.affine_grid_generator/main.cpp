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
        
        // Need at least 2 bytes for theta tensor creation
        if (Size < 2) {
            return 0;
        }
        
        // Create theta tensor (transformation matrix)
        torch::Tensor theta = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have enough data for the remaining parameters
        if (offset + 4 > Size) {
            return 0;
        }
        
        // Parse size parameters
        int64_t N = static_cast<int64_t>(Data[offset++]) % 16;
        int64_t C = static_cast<int64_t>(Data[offset++]) % 16;
        int64_t H = static_cast<int64_t>(Data[offset++]) % 32;
        int64_t W = static_cast<int64_t>(Data[offset++]) % 32;
        
        // Parse align_corners parameter
        bool align_corners = false;
        if (offset < Size) {
            align_corners = Data[offset++] & 1;
        }
        
        // Try different scenarios for theta tensor
        try {
            // Attempt to use the tensor as is
            auto grid1 = torch::affine_grid_generator(theta, {N, C, H, W}, align_corners);
        } catch (const std::exception&) {
            // If that fails, try reshaping theta to valid dimensions for affine_grid_generator
            try {
                // For 2D: theta should be [N, 2, 3]
                if (theta.dim() != 3 || theta.size(1) != 2 || theta.size(2) != 3) {
                    // Reshape to [N, 2, 3] if possible
                    if (theta.numel() >= N * 2 * 3 && N > 0) {
                        theta = theta.reshape({N, 2, 3});
                        auto grid2 = torch::affine_grid_generator(theta, {N, C, H, W}, align_corners);
                    }
                }
            } catch (const std::exception&) {
                // Try 3D case: theta should be [N, 3, 4]
                try {
                    if (theta.numel() >= N * 3 * 4 && N > 0) {
                        theta = theta.reshape({N, 3, 4});
                        auto grid3 = torch::affine_grid_generator(theta, {N, C, H, W, W}, align_corners);
                    }
                } catch (const std::exception&) {
                    // If all attempts fail, create a valid theta tensor
                    if (N > 0) {
                        // Try 2D case
                        theta = torch::rand({N, 2, 3});
                        auto grid4 = torch::affine_grid_generator(theta, {N, C, H, W}, align_corners);
                        
                        // Try 3D case
                        theta = torch::rand({N, 3, 4});
                        auto grid5 = torch::affine_grid_generator(theta, {N, C, H, W, W}, align_corners);
                    }
                }
            }
        }
        
        // Try edge cases
        if (offset < Size) {
            uint8_t edge_case = Data[offset++] % 5;
            
            switch (edge_case) {
                case 0: {
                    // Zero batch size
                    if (theta.dim() == 3 && theta.size(1) == 2 && theta.size(2) == 3) {
                        try {
                            auto grid6 = torch::affine_grid_generator(theta, {0, C, H, W}, align_corners);
                        } catch (const std::exception&) {}
                    }
                    break;
                }
                case 1: {
                    // Zero height/width
                    if (theta.dim() == 3 && theta.size(1) == 2 && theta.size(2) == 3 && theta.size(0) > 0) {
                        try {
                            auto grid7 = torch::affine_grid_generator(theta, {theta.size(0), C, 0, W}, align_corners);
                        } catch (const std::exception&) {}
                        
                        try {
                            auto grid8 = torch::affine_grid_generator(theta, {theta.size(0), C, H, 0}, align_corners);
                        } catch (const std::exception&) {}
                    }
                    break;
                }
                case 2: {
                    // Mismatch between theta batch size and size parameter batch size
                    if (theta.dim() == 3 && theta.size(0) > 1) {
                        try {
                            auto grid9 = torch::affine_grid_generator(theta, {theta.size(0) - 1, C, H, W}, align_corners);
                        } catch (const std::exception&) {}
                    }
                    break;
                }
                case 3: {
                    // Identity transformation
                    if (N > 0) {
                        theta = torch::zeros({N, 2, 3});
                        theta.select(1, 0).select(1, 0).fill_(1);
                        theta.select(1, 1).select(1, 1).fill_(1);
                        auto grid10 = torch::affine_grid_generator(theta, {N, C, H, W}, align_corners);
                    }
                    break;
                }
                case 4: {
                    // Extreme values in theta
                    if (N > 0) {
                        theta = torch::ones({N, 2, 3}) * 1e10;
                        try {
                            auto grid11 = torch::affine_grid_generator(theta, {N, C, H, W}, align_corners);
                        } catch (const std::exception&) {}
                        
                        theta = torch::ones({N, 2, 3}) * -1e10;
                        try {
                            auto grid12 = torch::affine_grid_generator(theta, {N, C, H, W}, align_corners);
                        } catch (const std::exception&) {}
                    }
                    break;
                }
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
