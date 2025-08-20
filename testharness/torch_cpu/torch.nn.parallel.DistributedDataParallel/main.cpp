#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create a simple model
        torch::nn::Linear model(10, 10);
        
        // Parse parameters for model testing
        uint8_t broadcast_buffers_byte = Data[offset++];
        bool broadcast_buffers = broadcast_buffers_byte % 2 == 1;
        
        uint8_t find_unused_parameters_byte = Data[offset++];
        bool find_unused_parameters = find_unused_parameters_byte % 2 == 1;
        
        uint8_t check_reduction_byte = Data[offset++];
        bool check_reduction = check_reduction_byte % 2 == 1;
        
        uint8_t gradient_as_bucket_view_byte = Data[offset++];
        bool gradient_as_bucket_view = gradient_as_bucket_view_byte % 2 == 1;
        
        // Create input tensor for the model
        torch::Tensor input;
        if (offset < Size) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            input = torch::randn({2, 10});
        }
        
        // Test basic model functionality
        try {
            auto output = model->forward(input);
            
            // Try backward pass if forward succeeded
            try {
                if (output.dim() > 0 && output.size(0) > 0) {
                    auto target = torch::ones_like(output);
                    auto loss = torch::mse_loss(output, target);
                    loss.backward();
                }
            } catch (...) {
                // Backward might fail, that's ok for fuzzing
            }
        } catch (...) {
            // Forward might fail, that's ok for fuzzing
        }
        
        // Try with device_ids parameter simulation
        try {
            std::vector<int64_t> device_ids;
            if (offset + 1 < Size) {
                uint8_t num_devices = Data[offset++] % 4;  // Limit to reasonable number
                for (uint8_t i = 0; i < num_devices && offset < Size; i++) {
                    device_ids.push_back(static_cast<int64_t>(Data[offset++]) % 8);  // Limit to reasonable device IDs
                }
            }
            
            // Test model with different configurations
            auto output = model->forward(input);
            if (output.dim() > 0 && output.size(0) > 0) {
                auto target = torch::ones_like(output);
                auto loss = torch::mse_loss(output, target);
                loss.backward();
            }
        } catch (...) {
            // Operations might fail, that's ok for fuzzing
        }
        
        // Try with output_device parameter simulation
        try {
            int64_t output_device = 0;
            if (offset < Size) {
                output_device = static_cast<int64_t>(Data[offset++]) % 8;  // Limit to reasonable device ID
            }
            
            // Test model forward and backward
            try {
                auto output = model->forward(input);
                if (output.dim() > 0 && output.size(0) > 0) {
                    auto target = torch::ones_like(output);
                    auto loss = torch::mse_loss(output, target);
                    loss.backward();
                }
            } catch (...) {
                // Operations might fail, that's ok for fuzzing
            }
        } catch (...) {
            // Model operations might fail, that's ok for fuzzing
        }
        
        // Try with a more complex model
        try {
            torch::nn::Sequential complex_model(
                torch::nn::Linear(10, 20),
                torch::nn::ReLU(),
                torch::nn::Linear(20, 10),
                torch::nn::Sigmoid()
            );
            
            // Try forward and backward
            try {
                auto output = complex_model->forward(input);
                if (output.dim() > 0 && output.size(0) > 0) {
                    auto target = torch::ones_like(output);
                    auto loss = torch::mse_loss(output, target);
                    loss.backward();
                }
            } catch (...) {
                // Operations might fail, that's ok for fuzzing
            }
        } catch (...) {
            // Model creation might fail, that's ok for fuzzing
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}