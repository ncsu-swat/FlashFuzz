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
        
        // Need at least a few bytes to create meaningful input
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a list of devices (use CPU for fuzzing)
        std::vector<torch::Device> devices;
        
        // Determine number of devices (1-4) based on available data
        uint8_t num_devices = 1;
        if (offset < Size) {
            num_devices = (Data[offset++] % 4) + 1;
        }
        
        // Add CPU devices
        for (uint8_t i = 0; i < num_devices; i++) {
            devices.push_back(torch::Device(torch::kCPU));
        }
        
        // Determine chunk_size (optional parameter)
        int64_t chunk_size = -1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&chunk_size, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Determine dim (optional parameter)
        int64_t dim = 0;
        if (offset < Size) {
            dim = static_cast<int64_t>(Data[offset++]);
        }
        
        // Call scatter with different parameter combinations
        std::vector<torch::Tensor> scattered_tensors;
        
        // Try different combinations of parameters based on available data
        if (offset < Size) {
            uint8_t param_choice = Data[offset++] % 4;
            
            switch (param_choice) {
                case 0:
                    // Basic scatter
                    scattered_tensors = torch::scatter(input_tensor, devices);
                    break;
                case 1:
                    // Scatter with dim
                    scattered_tensors = torch::scatter(input_tensor, devices, dim);
                    break;
                case 2:
                    // Scatter with chunk_size
                    scattered_tensors = torch::scatter(input_tensor, devices, chunk_size);
                    break;
                case 3:
                    // Scatter with both dim and chunk_size
                    scattered_tensors = torch::scatter(input_tensor, devices, chunk_size, dim);
                    break;
            }
        } else {
            // Default to basic scatter if no more data
            scattered_tensors = torch::scatter(input_tensor, devices);
        }
        
        // Verify the result is not empty
        if (scattered_tensors.empty()) {
            return 0;
        }
        
        // Try to access elements of the scattered tensors to ensure they're valid
        for (const auto& tensor : scattered_tensors) {
            if (tensor.defined()) {
                auto sizes = tensor.sizes();
                auto dtype = tensor.dtype();
                auto numel = tensor.numel();
                
                // If tensor has elements, try to access one
                if (numel > 0) {
                    auto item = tensor.flatten()[0];
                }
            }
        }
        
        // Try scatter with a list of tensors
        if (offset < Size && Size - offset > 4) {
            std::vector<torch::Tensor> tensor_list;
            
            // Create 1-3 additional tensors
            uint8_t num_tensors = (Data[offset++] % 3) + 1;
            for (uint8_t i = 0; i < num_tensors && offset < Size; i++) {
                tensor_list.push_back(fuzzer_utils::createTensor(Data, Size, offset));
            }
            
            // Scatter the list of tensors
            if (!tensor_list.empty()) {
                auto scattered_lists = torch::scatter(tensor_list, devices);
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
