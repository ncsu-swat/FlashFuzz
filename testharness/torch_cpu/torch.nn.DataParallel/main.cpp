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
        
        // Check if we have enough data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create a model to wrap with DataParallel
        torch::nn::Linear model = nullptr;
        
        // Parse input features and output features from the data
        uint16_t in_features = 0;
        uint16_t out_features = 0;
        
        if (offset + sizeof(uint16_t) <= Size) {
            std::memcpy(&in_features, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
        }
        
        if (offset + sizeof(uint16_t) <= Size) {
            std::memcpy(&out_features, Data + offset, sizeof(uint16_t));
            offset += sizeof(uint16_t);
        }
        
        // Ensure we have at least 1 feature for input and output
        in_features = (in_features % 100) + 1;
        out_features = (out_features % 100) + 1;
        
        // Create a simple linear model
        model = torch::nn::Linear(in_features, out_features);
        
        // Create a DataParallel wrapper around the model
        auto data_parallel_model = torch::nn::parallel::DataParallel(model);
        
        // Create input tensor for the model
        torch::Tensor input;
        try {
            input = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If tensor creation fails, create a default tensor
            input = torch::randn({2, in_features});
        }
        
        // Ensure input has correct shape for the model
        if (input.dim() < 2 || input.size(input.dim() - 1) != in_features) {
            // Reshape the tensor to have the correct last dimension
            std::vector<int64_t> new_shape;
            int64_t total_elements = input.numel();
            
            if (total_elements == 0) {
                // Handle empty tensor case
                new_shape = {1, in_features};
                input = torch::zeros(new_shape);
            } else {
                // Calculate batch size
                int64_t batch_size = std::max<int64_t>(1, total_elements / in_features);
                new_shape = {batch_size, in_features};
                
                // Reshape or create new tensor
                if (input.numel() >= batch_size * in_features) {
                    input = input.reshape(new_shape);
                } else {
                    input = torch::randn(new_shape);
                }
            }
        }
        
        // Parse device_ids parameter
        std::vector<int64_t> device_ids;
        uint8_t num_devices = 0;
        
        if (offset < Size) {
            num_devices = Data[offset++] % 4;  // Limit to 0-3 devices
            
            for (uint8_t i = 0; i < num_devices && offset + sizeof(int64_t) <= Size; ++i) {
                int64_t device_id;
                std::memcpy(&device_id, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                
                // Ensure device_id is non-negative
                device_id = std::abs(device_id) % 8;  // Limit to 0-7
                device_ids.push_back(device_id);
            }
        }
        
        // Parse output_device parameter
        int64_t output_device = -1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&output_device, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure output_device is valid
            if (output_device < 0) {
                output_device = -1;  // Default value
            } else {
                output_device = output_device % 8;  // Limit to 0-7
            }
        }
        
        // Parse dim parameter
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Create DataParallel with the parsed parameters
        if (!device_ids.empty()) {
            data_parallel_model = torch::nn::parallel::DataParallel(
                model, 
                torch::nn::parallel::DataParallelOptions()
                    .device_ids(device_ids)
                    .output_device(output_device)
                    .dim(dim)
            );
        }
        
        // Forward pass through the DataParallel model
        torch::Tensor output;
        try {
            output = data_parallel_model->forward(input);
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected in some cases
            return 0;
        }
        
        // Test some operations on the output
        if (output.defined() && output.numel() > 0) {
            auto sum = output.sum();
            auto mean = output.mean();
            auto max_val = output.max();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
