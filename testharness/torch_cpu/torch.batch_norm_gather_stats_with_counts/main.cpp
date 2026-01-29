#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with result

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
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 16) {
            return 0;
        }
        
        // Parse dimensions from fuzz data
        uint8_t num_batches = (Data[offset] % 4) + 1;  // 1-4 batches
        offset++;
        uint8_t num_features = (Data[offset] % 8) + 1; // 1-8 features
        offset++;
        
        // Create input tensor - shape doesn't matter much for this gather operation
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        if (input.numel() == 0) {
            input = torch::randn({2, static_cast<int64_t>(num_features), 4, 4});
        }
        
        // Ensure input is float type for batch norm operations
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Create mean tensor - shape (num_batches, num_features)
        // These represent per-batch statistics gathered from multiple workers
        torch::Tensor mean = torch::randn({static_cast<int64_t>(num_batches), 
                                           static_cast<int64_t>(num_features)});
        
        // Create invstd tensor - same shape as mean, must be positive
        torch::Tensor invstd = torch::rand({static_cast<int64_t>(num_batches), 
                                            static_cast<int64_t>(num_features)}) + 0.1;
        
        // Create running_mean and running_var - shape (num_features,)
        torch::Tensor running_mean = torch::zeros({static_cast<int64_t>(num_features)});
        torch::Tensor running_var = torch::ones({static_cast<int64_t>(num_features)});
        
        // Parse momentum from fuzz data
        double momentum = 0.1;
        if (offset + sizeof(float) <= Size) {
            float raw_momentum;
            std::memcpy(&raw_momentum, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Clamp momentum to valid range [0, 1]
            if (std::isfinite(raw_momentum)) {
                momentum = std::fmod(std::fabs(raw_momentum), 1.0);
            }
        }
        
        // Parse eps from fuzz data
        double eps = 1e-5;
        if (offset + sizeof(float) <= Size) {
            float raw_eps;
            std::memcpy(&raw_eps, Data + offset, sizeof(float));
            offset += sizeof(float);
            // Ensure eps is positive and reasonable
            if (std::isfinite(raw_eps) && raw_eps > 0) {
                eps = std::fmin(std::fabs(raw_eps), 1.0);
            }
        }
        
        // Create counts tensor - shape (num_batches,), representing element count per batch
        // These must be positive integers representing how many elements each batch had
        torch::Tensor counts = torch::randint(1, 1000, {static_cast<int64_t>(num_batches)}, 
                                               torch::kFloat32);
        
        // Vary the tensor values based on fuzz input to improve coverage
        if (offset < Size) {
            float scale = static_cast<float>(Data[offset]) / 255.0f * 10.0f;
            offset++;
            mean = mean * scale;
        }
        
        if (offset < Size) {
            float invstd_scale = static_cast<float>(Data[offset]) / 255.0f + 0.01f;
            offset++;
            invstd = invstd * invstd_scale;
        }
        
        // Try different combinations of running stats being defined/undefined
        bool use_running_mean = true;
        bool use_running_var = true;
        if (offset < Size) {
            use_running_mean = (Data[offset] & 0x01) != 0;
            use_running_var = (Data[offset] & 0x02) != 0;
            offset++;
        }
        
        try {
            // Apply batch_norm_gather_stats_with_counts
            auto result = torch::batch_norm_gather_stats_with_counts(
                input, 
                mean, 
                invstd, 
                use_running_mean ? running_mean : torch::Tensor(),
                use_running_var ? running_var : torch::Tensor(),
                momentum, 
                eps, 
                counts);
            
            // Use the result to prevent optimization
            auto mean_result = std::get<0>(result);
            auto var_result = std::get<1>(result);
            
            // Perform some operation on the result to ensure it's used
            if (mean_result.defined() && mean_result.numel() > 0) {
                volatile float sum1 = mean_result.sum().item<float>();
                (void)sum1;
            }
            if (var_result.defined() && var_result.numel() > 0) {
                volatile float sum2 = var_result.sum().item<float>();
                (void)sum2;
            }
        }
        catch (const c10::Error &e) {
            // Expected errors from shape mismatches, etc. - silently catch
        }
        catch (const std::runtime_error &e) {
            // Expected runtime errors - silently catch
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}