#include "fuzzer_utils.h"
#include <iostream>
#include <cmath>

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
        
        if (Size < 8) {
            return 0;
        }
        
        // Extract parameters for BatchNorm1d from the data
        double eps = 1e-5;
        double momentum = 0.1;
        bool affine = true;
        bool track_running_stats = true;
        
        // Extract eps parameter
        if (offset + 4 <= Size) {
            uint32_t eps_raw;
            std::memcpy(&eps_raw, Data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            // Map to reasonable eps range [1e-10, 1e-1]
            eps = 1e-10 + (static_cast<double>(eps_raw) / UINT32_MAX) * (1e-1 - 1e-10);
        }
        
        // Extract momentum parameter
        if (offset + 4 <= Size) {
            uint32_t momentum_raw;
            std::memcpy(&momentum_raw, Data + offset, sizeof(uint32_t));
            offset += sizeof(uint32_t);
            // Map to [0, 1] range
            momentum = static_cast<double>(momentum_raw) / UINT32_MAX;
        }
        
        // Extract boolean parameters
        if (offset < Size) {
            affine = (Data[offset] % 2) == 1;
            offset++;
        }
        
        if (offset < Size) {
            track_running_stats = (Data[offset] % 2) == 1;
            offset++;
        }
        
        // Create input tensor - BatchNorm1d expects 2D (N, C) or 3D (N, C, L)
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Reshape input to valid dimensions for BatchNorm1d
        // BatchNorm1d expects input of shape (N, C) or (N, C, L)
        int64_t total_elements = input.numel();
        if (total_elements < 2) {
            return 0;
        }
        
        // Determine batch size and channels
        int64_t batch_size = 1;
        int64_t channels = 1;
        int64_t length = 1;
        bool use_3d = false;
        
        if (offset < Size) {
            // Use fuzzer data to determine shape configuration
            uint8_t shape_config = Data[offset % Size];
            offset++;
            
            if (shape_config % 3 == 0) {
                // 2D: (N, C) - flatten everything
                batch_size = std::max(int64_t(1), total_elements / 4);
                channels = total_elements / batch_size;
                if (channels < 1) channels = 1;
                int64_t new_total = batch_size * channels;
                input = input.flatten().narrow(0, 0, std::min(new_total, total_elements))
                            .reshape({batch_size, channels});
                use_3d = false;
            } else if (shape_config % 3 == 1) {
                // 3D: (N, C, L)
                batch_size = std::max(int64_t(1), (int64_t)(std::cbrt(total_elements)));
                channels = std::max(int64_t(1), (int64_t)(std::sqrt(total_elements / batch_size)));
                length = std::max(int64_t(1), total_elements / (batch_size * channels));
                int64_t new_total = batch_size * channels * length;
                if (new_total > total_elements) {
                    length = total_elements / (batch_size * channels);
                    if (length < 1) {
                        channels = total_elements / batch_size;
                        length = 1;
                    }
                }
                new_total = batch_size * channels * length;
                try {
                    input = input.flatten().narrow(0, 0, std::min(new_total, total_elements))
                                .reshape({batch_size, channels, length});
                    use_3d = true;
                } catch (...) {
                    // Fall back to 2D
                    new_total = batch_size * channels;
                    input = input.flatten().narrow(0, 0, std::min(new_total, total_elements))
                                .reshape({batch_size, channels});
                    use_3d = false;
                }
            } else {
                // Simple 2D with small batch
                batch_size = 2;
                channels = total_elements / batch_size;
                if (channels < 1) {
                    batch_size = 1;
                    channels = total_elements;
                }
                int64_t new_total = batch_size * channels;
                input = input.flatten().narrow(0, 0, std::min(new_total, total_elements))
                            .reshape({batch_size, channels});
                use_3d = false;
            }
        } else {
            // Default: 2D shape
            batch_size = std::min(int64_t(4), total_elements);
            channels = total_elements / batch_size;
            if (channels < 1) channels = 1;
            int64_t new_total = batch_size * channels;
            input = input.flatten().narrow(0, 0, std::min(new_total, total_elements))
                        .reshape({batch_size, channels});
            use_3d = false;
        }
        
        // Ensure input is float type for batch norm
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Get number of channels from reshaped input
        int64_t num_features = input.size(1);
        if (num_features < 1) {
            return 0;
        }
        
        // Create the BatchNorm1d module
        // Note: LazyBatchNorm1d is Python-only, use BatchNorm1d in C++
        torch::nn::BatchNorm1d batch_norm(
            torch::nn::BatchNorm1dOptions(num_features)
                .eps(eps)
                .momentum(momentum)
                .affine(affine)
                .track_running_stats(track_running_stats)
        );
        
        // Apply the batch norm operation
        torch::Tensor output;
        try {
            output = batch_norm->forward(input);
        } catch (const c10::Error&) {
            // Shape or dimension mismatch - valid rejection
            return 0;
        }
        
        // Force materialization of the tensor
        output = output.clone();
        
        // Verify output properties
        auto sizes = output.sizes();
        (void)sizes;
        
        // Access running stats if they're being tracked
        if (track_running_stats && batch_norm->running_mean.defined()) {
            auto mean_sum = batch_norm->running_mean.sum().item<float>();
            (void)mean_sum;
        }
        if (track_running_stats && batch_norm->running_var.defined()) {
            auto var_sum = batch_norm->running_var.sum().item<float>();
            (void)var_sum;
        }
        
        // Access learnable parameters if affine is true
        if (affine && batch_norm->weight.defined()) {
            auto weight_sum = batch_norm->weight.sum().item<float>();
            (void)weight_sum;
        }
        if (affine && batch_norm->bias.defined()) {
            auto bias_sum = batch_norm->bias.sum().item<float>();
            (void)bias_sum;
        }
        
        // Test in eval mode as well
        batch_norm->eval();
        try {
            torch::Tensor eval_output = batch_norm->forward(input);
            eval_output = eval_output.clone();
        } catch (const c10::Error&) {
            // May fail in eval mode without running stats
        }
        
        // Test reset_running_stats if tracking
        if (track_running_stats) {
            try {
                batch_norm->reset_running_stats();
            } catch (const c10::Error&) {
                // May not be available
            }
        }
        
        // Test with different input having same channel dimension
        if (use_3d) {
            // Create another 3D input with same channels but different batch/length
            int64_t new_batch = std::max(int64_t(1), batch_size / 2 + 1);
            int64_t new_length = std::max(int64_t(1), length * 2);
            torch::Tensor new_input = torch::randn({new_batch, num_features, new_length});
            try {
                batch_norm->train();
                torch::Tensor new_output = batch_norm->forward(new_input);
                (void)new_output;
            } catch (const c10::Error&) {
                // Shape mismatch is acceptable
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