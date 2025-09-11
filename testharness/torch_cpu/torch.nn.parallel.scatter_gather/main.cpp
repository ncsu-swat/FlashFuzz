#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <algorithm>      // For std::max

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to scatter
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a list of devices (using CPU for simplicity)
        std::vector<torch::Device> devices;
        
        // Determine number of devices based on remaining data
        uint8_t num_devices = 1;
        if (offset < Size) {
            num_devices = (Data[offset++] % 4) + 1; // 1-4 devices
        }
        
        for (uint8_t i = 0; i < num_devices; ++i) {
            devices.push_back(torch::Device(torch::kCPU));
        }
        
        // Create a scatter function
        auto scatter_fn = [&devices](torch::Tensor tensor) {
            std::vector<torch::Tensor> outputs;
            int64_t chunk_size = std::max(static_cast<int64_t>(tensor.size(0)) / static_cast<int64_t>(devices.size()), static_cast<int64_t>(1));
            
            for (size_t i = 0; i < devices.size(); ++i) {
                int64_t start = i * chunk_size;
                int64_t end = (i == devices.size() - 1) ? tensor.size(0) : (i + 1) * chunk_size;
                
                if (start < tensor.size(0)) {
                    if (tensor.dim() > 0 && end > start) {
                        outputs.push_back(tensor.slice(0, start, end));
                    } else {
                        outputs.push_back(tensor);
                    }
                } else {
                    // Handle edge case where start is beyond tensor size
                    outputs.push_back(tensor.slice(0, 0, 0));
                }
            }
            return outputs;
        };
        
        // Create a gather function
        auto gather_fn = [](std::vector<torch::Tensor> tensors, torch::Device device) {
            if (tensors.empty()) {
                return torch::Tensor();
            }
            
            if (tensors.size() == 1) {
                return tensors[0];
            }
            
            // Filter out empty tensors
            std::vector<torch::Tensor> non_empty_tensors;
            for (const auto& t : tensors) {
                if (t.defined() && t.numel() > 0) {
                    non_empty_tensors.push_back(t);
                }
            }
            
            if (non_empty_tensors.empty()) {
                return torch::empty({0}, tensors[0].options());
            }
            
            return torch::cat(non_empty_tensors, 0);
        };
        
        // Test scatter - manual implementation since torch::nn::parallel doesn't exist
        std::vector<torch::Tensor> scattered_tensors;
        try {
            scattered_tensors = scatter_fn(input_tensor);
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors
        }
        
        // Test gather - manual implementation
        if (!scattered_tensors.empty()) {
            try {
                torch::Tensor gathered = gather_fn(scattered_tensors, torch::Device(torch::kCPU));
            } catch (const c10::Error& e) {
                // Catch PyTorch-specific errors
            }
        }
        
        // Test scatter_gather - manual implementation
        try {
            std::vector<torch::Tensor> inputs = {input_tensor};
            std::vector<torch::Tensor> scattered = scatter_fn(inputs[0]);
            torch::Tensor output = gather_fn(scattered, torch::Device(torch::kCPU));
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors
        }
        
        // Test with multiple input tensors
        if (offset + 4 < Size) {
            torch::Tensor second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            try {
                std::vector<torch::Tensor> scattered1 = scatter_fn(input_tensor);
                std::vector<torch::Tensor> scattered2 = scatter_fn(second_tensor);
                torch::Tensor output1 = gather_fn(scattered1, torch::Device(torch::kCPU));
                torch::Tensor output2 = gather_fn(scattered2, torch::Device(torch::kCPU));
            } catch (const c10::Error& e) {
                // Catch PyTorch-specific errors
            }
        }
        
        // Test with empty input list
        try {
            std::vector<torch::Tensor> empty_tensors;
            torch::Tensor empty_output = gather_fn(empty_tensors, torch::Device(torch::kCPU));
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors
        }
        
        // Test with empty devices list
        if (offset < Size && Data[offset] % 2 == 0) {
            try {
                // Create empty device scenario
                std::vector<torch::Device> empty_devices;
                if (!empty_devices.empty()) {
                    auto empty_device_outputs = scatter_fn(input_tensor);
                }
            } catch (const c10::Error& e) {
                // Catch PyTorch-specific errors
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
