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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test torch.tensor functionality by cloning the existing tensor
        torch::Tensor copied_tensor = tensor.clone();
        
        // Test creating a tensor from a scalar value
        if (offset + sizeof(float) <= Size) {
            float scalar_value;
            std::memcpy(&scalar_value, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            torch::Tensor scalar_tensor = torch::tensor(scalar_value);
        }
        
        // Test creating a tensor from a vector
        if (offset + 4 <= Size) {
            std::vector<int> vec_data;
            for (size_t i = 0; i < 4 && offset < Size; i++) {
                vec_data.push_back(static_cast<int>(Data[offset++]));
            }
            
            torch::Tensor vec_tensor = torch::tensor(vec_data);
        }
        
        // Test creating a tensor with specific options
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            
            torch::Tensor options_tensor = tensor.to(dtype);
        }
        
        // Test creating a tensor from a 2D vector
        if (offset + 4 <= Size) {
            std::vector<std::vector<int>> vec_2d_data;
            vec_2d_data.push_back(std::vector<int>());
            vec_2d_data.push_back(std::vector<int>());
            
            for (size_t i = 0; i < 4 && offset < Size; i++) {
                vec_2d_data[i % 2].push_back(static_cast<int>(Data[offset++]));
            }
            
            torch::Tensor vec_2d_tensor = torch::tensor(vec_2d_data);
        }
        
        // Test creating a tensor with requires_grad option
        if (offset < Size) {
            bool requires_grad = Data[offset++] % 2 == 0;
            torch::Tensor grad_tensor = tensor.clone().requires_grad_(requires_grad);
        }
        
        // Test creating a tensor with pin_memory option
        if (offset < Size) {
            bool pin_memory = Data[offset++] % 2 == 0;
            torch::Tensor pinned_tensor = tensor.clone().pin_memory(pin_memory);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
