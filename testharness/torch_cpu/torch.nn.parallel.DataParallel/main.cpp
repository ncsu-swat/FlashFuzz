#include "fuzzer_utils.h"
#include <iostream>
#include <torch/torch.h>

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
        
        // Skip if data is too small
        if (Size < 8) {
            return 0;
        }
        
        // Define a simple model for testing
        struct SimpleModel : torch::nn::Module {
            SimpleModel() {
                linear = register_module("linear", torch::nn::Linear(10, 5));
            }
            
            torch::Tensor forward(torch::Tensor x) {
                return linear->forward(x);
            }
            
            torch::nn::Linear linear{nullptr};
        };
        
        auto model = std::make_shared<SimpleModel>();
        
        // Parse batch size and other parameters from input data
        uint8_t batch_size = (Data[offset++] % 8) + 1;  // 1-8 batch size
        int64_t dim = static_cast<int64_t>(Data[offset++] % 2);  // dim 0 or 1
        
        // Create input tensor with proper shape for the model [batch_size, 10]
        torch::Tensor input = torch::randn({batch_size, 10});
        
        // Use remaining data to perturb the input values
        if (offset < Size) {
            torch::Tensor fuzz_data = fuzzer_utils::createTensor(Data, Size, offset);
            // Flatten and use first elements to modify input
            fuzz_data = fuzz_data.flatten();
            int64_t copy_len = std::min(fuzz_data.numel(), input.numel());
            if (copy_len > 0 && fuzz_data.scalar_type() == input.scalar_type()) {
                try {
                    input.flatten().slice(0, 0, copy_len).copy_(fuzz_data.slice(0, 0, copy_len));
                } catch (...) {
                    // Ignore copy errors, use random input
                }
            }
        }
        
        // Check if CUDA is available
        if (!torch::cuda::is_available()) {
            // On CPU-only systems, we can still test the API path but it will fail
            // This is expected behavior - the API requires CUDA
            // Just test that the model works on CPU directly
            try {
                torch::Tensor output = model->forward(input);
                (void)output;
            } catch (...) {
                // Expected on some configurations
            }
            return 0;
        }
        
        // CUDA is available - test data_parallel functionality
        int64_t num_devices = torch::cuda::device_count();
        if (num_devices == 0) {
            return 0;
        }
        
        // Build device_ids list from available devices
        std::vector<int64_t> device_ids;
        uint8_t num_requested = (Data[offset % Size] % static_cast<uint8_t>(num_devices)) + 1;
        for (int64_t i = 0; i < num_requested && i < num_devices; ++i) {
            device_ids.push_back(i);
        }
        
        // Move model and input to CUDA
        model->to(torch::kCUDA);
        input = input.to(torch::kCUDA);
        
        // Test 1: Basic data_parallel call with device_ids
        try {
            torch::Tensor output = torch::nn::parallel::data_parallel(
                model,
                input,
                device_ids
            );
            (void)output;
        } catch (...) {
            // May fail due to device configuration
        }
        
        // Test 2: data_parallel with output_device specified
        if (offset + 1 < Size) {
            int64_t output_device = static_cast<int64_t>(Data[++offset % Size]) % num_devices;
            try {
                torch::Tensor output = torch::nn::parallel::data_parallel(
                    model,
                    input,
                    device_ids,
                    output_device
                );
                (void)output;
            } catch (...) {
                // May fail due to device configuration
            }
        }
        
        // Test 3: data_parallel with dim parameter
        try {
            torch::Tensor output = torch::nn::parallel::data_parallel(
                model,
                input,
                device_ids,
                std::nullopt,  // output_device
                dim
            );
            (void)output;
        } catch (...) {
            // May fail due to invalid dim
        }
        
        // Test 4: Single device (should work like regular forward)
        try {
            std::vector<int64_t> single_device = {0};
            torch::Tensor output = torch::nn::parallel::data_parallel(
                model,
                input,
                single_device
            );
            (void)output;
        } catch (...) {
            // May fail
        }
        
        // Test 5: No device_ids (use default)
        try {
            torch::Tensor output = torch::nn::parallel::data_parallel(
                model,
                input
            );
            (void)output;
        } catch (...) {
            // May fail
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}