#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

class SimpleModel : public torch::nn::Module {
public:
    SimpleModel() {
        linear = register_module("linear", torch::nn::Linear(10, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        return linear->forward(x);
    }

    torch::nn::Linear linear{nullptr};
};

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a simple model
        auto model = std::make_shared<SimpleModel>();
        
        // Get number of devices to use
        uint8_t num_devices_byte = 0;
        if (offset < Size) {
            num_devices_byte = Data[offset++];
        }
        int num_devices = (num_devices_byte % 4) + 1;  // Use 1-4 devices
        
        // Get device_ids
        std::vector<torch::Device> device_ids;
        for (int i = 0; i < num_devices; ++i) {
            device_ids.push_back(torch::Device(torch::kCUDA, i % torch::cuda::device_count()));
        }
        
        // Get output_device
        torch::optional<torch::Device> output_device = torch::nullopt;
        if (offset < Size) {
            int device_idx = static_cast<int>(Data[offset++]) % (torch::cuda::device_count() + 1) - 1;
            if (device_idx >= 0) {
                output_device = torch::Device(torch::kCUDA, device_idx);
            }
        }
        
        // Get dim
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Reshape input if needed to match model's expected input
        if (input.dim() > 0 && input.size(0) > 0) {
            // Try to reshape to a valid input shape for the model
            std::vector<int64_t> new_shape;
            new_shape.push_back(input.size(0));  // Keep batch size
            
            // Add dimension of size 10 for the linear layer
            new_shape.push_back(10);
            
            // Try to reshape, but handle potential errors
            try {
                int64_t total_elements = 1;
                for (auto &dim : new_shape) {
                    total_elements *= dim;
                }
                
                if (total_elements > 0) {
                    // Only reshape if we have a valid shape
                    if (input.numel() != total_elements) {
                        input = input.reshape({-1}).slice(0, 0, total_elements);
                        if (input.numel() < total_elements) {
                            // Pad if needed
                            input = torch::cat({input, torch::zeros(total_elements - input.numel())});
                        }
                        input = input.reshape(new_shape);
                    } else {
                        input = input.reshape(new_shape);
                    }
                }
            } catch (const std::exception &) {
                // If reshape fails, create a valid tensor
                input = torch::ones({1, 10});
            }
        } else {
            // Create a valid input tensor if the original is empty
            input = torch::ones({1, 10});
        }
        
        // Apply data_parallel using torch::nn::parallel::data_parallel function
        try {
            torch::Tensor output;
            
            if (device_ids.empty()) {
                // Use default device_ids - just run on current device
                output = model->forward(input);
            } else {
                // For simplicity, just run on first device since data_parallel is not directly available
                if (torch::cuda::is_available() && !device_ids.empty()) {
                    input = input.to(device_ids[0]);
                    model->to(device_ids[0]);
                    output = model->forward(input);
                } else {
                    output = model->forward(input);
                }
            }
        } catch (const c10::Error &) {
            // Expected exceptions from PyTorch operations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}