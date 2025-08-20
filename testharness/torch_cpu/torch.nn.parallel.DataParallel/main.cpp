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
        
        // Skip if data is too small
        if (Size < 4) {
            return 0;
        }
        
        // Create a simple model to wrap with DataParallel
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
        
        // Parse device IDs from input data
        uint8_t num_devices = Data[offset++] % 4 + 1; // 1-4 devices
        std::vector<torch::Device> device_ids;
        for (uint8_t i = 0; i < num_devices && offset < Size; ++i) {
            int device_id = static_cast<int>(Data[offset++]) % 8; // Device IDs 0-7
            device_ids.push_back(torch::Device(torch::kCUDA, device_id));
        }
        
        // Create DataParallel wrapper
        auto data_parallel_model = torch::nn::parallel::data_parallel(model, device_ids);
        
        // Create input tensor
        torch::Tensor input;
        if (offset < Size) {
            input = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure input has at least 2 dimensions for batch processing
            if (input.dim() < 2) {
                // Reshape to add batch dimension if needed
                std::vector<int64_t> new_shape;
                new_shape.push_back(1); // Batch size of 1
                for (int64_t i = 0; i < input.dim(); ++i) {
                    new_shape.push_back(input.size(i));
                }
                if (input.dim() == 0) {
                    new_shape.push_back(1); // Add feature dimension for scalar
                }
                input = input.reshape(new_shape);
            }
            
            // Ensure the last dimension is compatible with the model's input size
            if (input.size(-1) != 10) {
                // Resize the last dimension to match model input
                std::vector<int64_t> new_shape;
                for (int64_t i = 0; i < input.dim() - 1; ++i) {
                    new_shape.push_back(input.size(i));
                }
                new_shape.push_back(10);
                input = input.reshape(new_shape);
            }
            
            // Apply the model
            torch::Tensor output = data_parallel_model(input);
        }
        
        // Test with different output_device settings
        if (offset < Size) {
            int output_device = static_cast<int>(Data[offset++]) % 8;
            torch::Device output_dev(torch::kCUDA, output_device);
            auto data_parallel_model_with_options = torch::nn::parallel::data_parallel(model, device_ids, output_dev);
            
            if (input.defined()) {
                torch::Tensor output = data_parallel_model_with_options(input);
            }
        }
        
        // Test with dim parameter
        if (offset < Size) {
            int64_t dim = static_cast<int64_t>(Data[offset++]) % 4;
            auto data_parallel_model_with_dim = torch::nn::parallel::data_parallel(model, device_ids, c10::nullopt, dim);
            
            if (input.defined()) {
                torch::Tensor output = data_parallel_model_with_dim(input);
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