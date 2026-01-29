#include "fuzzer_utils.h"
#include <iostream>

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
        // cudnn_grid_sampler requires CUDA and cuDNN
        // On CPU-only builds, we cannot test this API
        if (!torch::cuda::is_available()) {
            // Skip silently on CPU - this API requires CUDA
            return 0;
        }

        // Need enough data to create meaningful tensors
        if (Size < 16) {
            return 0;
        }

        size_t offset = 0;

        // Read parameters from fuzzer data
        uint8_t batch_byte = (offset < Size) ? Data[offset++] : 1;
        uint8_t channels_byte = (offset < Size) ? Data[offset++] : 3;
        uint8_t h_byte = (offset < Size) ? Data[offset++] : 4;
        uint8_t w_byte = (offset < Size) ? Data[offset++] : 4;
        uint8_t h_out_byte = (offset < Size) ? Data[offset++] : 4;
        uint8_t w_out_byte = (offset < Size) ? Data[offset++] : 4;

        // Constrain dimensions to reasonable ranges
        int64_t N = (batch_byte % 4) + 1;      // 1-4
        int64_t C = (channels_byte % 16) + 1;  // 1-16
        int64_t H = (h_byte % 32) + 1;         // 1-32
        int64_t W = (w_byte % 32) + 1;         // 1-32
        int64_t H_out = (h_out_byte % 32) + 1; // 1-32
        int64_t W_out = (w_out_byte % 32) + 1; // 1-32

        // Create input tensor: [N, C, H, W] - must be float and contiguous
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        torch::Tensor input = torch::randn({N, C, H, W}, options).contiguous();

        // Create grid tensor: [N, H_out, W_out, 2]
        // Grid values should be in [-1, 1] for valid sampling coordinates
        torch::Tensor grid = torch::empty({N, H_out, W_out, 2}, options);
        
        // Fill grid with values from fuzzer data, normalized to [-1, 1]
        if (offset + N * H_out * W_out * 2 <= Size) {
            auto grid_accessor = grid.accessor<float, 4>();
            for (int64_t n = 0; n < N && offset < Size; n++) {
                for (int64_t h = 0; h < H_out && offset < Size; h++) {
                    for (int64_t w = 0; w < W_out && offset < Size; w++) {
                        // Normalize byte values to [-1, 1]
                        float x_val = (offset < Size) ? (Data[offset++] / 127.5f - 1.0f) : 0.0f;
                        float y_val = (offset < Size) ? (Data[offset++] / 127.5f - 1.0f) : 0.0f;
                        grid_accessor[n][h][w][0] = x_val;
                        grid_accessor[n][h][w][1] = y_val;
                    }
                }
            }
        } else {
            // Use random grid values in valid range
            grid = torch::rand({N, H_out, W_out, 2}, options) * 2.0f - 1.0f;
        }
        grid = grid.contiguous();

        // Call cudnn_grid_sampler
        // This requires cuDNN to be available
        torch::Tensor output = torch::cudnn_grid_sampler(input, grid);

        // Verify output and use it
        if (output.defined()) {
            // Output should be [N, C, H_out, W_out]
            auto sum = output.sum();
            // Force synchronization
            sum.item<float>();
        }
    }
    catch (const c10::Error &e)
    {
        // cuDNN or CUDA errors - expected in some configurations
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}