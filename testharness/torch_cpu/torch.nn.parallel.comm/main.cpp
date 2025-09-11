#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Get number of GPUs (or simulate multiple devices)
        int num_devices = 1;
        if (offset < Size) {
            num_devices = (Data[offset++] % 4) + 1; // 1 to 4 devices
        }
        
        // Create a list of devices
        std::vector<torch::Device> devices;
        for (int i = 0; i < num_devices; i++) {
            devices.push_back(torch::Device(torch::kCPU, i));
        }
        
        // Get chunk size
        int64_t chunk_size = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&chunk_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Test tensor operations that simulate parallel communication
        try {
            // Simulate broadcast by cloning tensor to multiple devices
            std::vector<torch::Tensor> broadcast_results;
            for (const auto& device : devices) {
                broadcast_results.push_back(input_tensor.to(device));
            }
        } catch (...) {
            // Ignore exceptions from broadcast simulation
        }
        
        // Test scatter simulation
        try {
            if (chunk_size > 0 && input_tensor.numel() > 0) {
                auto chunks = input_tensor.chunk(num_devices, 0);
            }
        } catch (...) {
            // Ignore exceptions from scatter simulation
        }
        
        // Test gather simulation
        try {
            std::vector<torch::Tensor> tensor_list;
            // Create a list of tensors to gather
            for (int i = 0; i < num_devices; i++) {
                if (offset < Size) {
                    tensor_list.push_back(fuzzer_utils::createTensor(Data, Size, offset));
                } else {
                    tensor_list.push_back(input_tensor.clone());
                }
            }
            if (!tensor_list.empty()) {
                auto result_gather = torch::cat(tensor_list, 0);
            }
        } catch (...) {
            // Ignore exceptions from gather simulation
        }
        
        // Test reduce_add simulation
        try {
            std::vector<torch::Tensor> tensor_list;
            // Create a list of tensors to reduce
            for (int i = 0; i < num_devices; i++) {
                if (offset < Size) {
                    tensor_list.push_back(fuzzer_utils::createTensor(Data, Size, offset));
                } else {
                    tensor_list.push_back(input_tensor.clone());
                }
            }
            if (!tensor_list.empty()) {
                torch::Tensor result_reduce = tensor_list[0];
                for (size_t i = 1; i < tensor_list.size(); i++) {
                    result_reduce = result_reduce + tensor_list[i];
                }
            }
        } catch (...) {
            // Ignore exceptions from reduce_add simulation
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
