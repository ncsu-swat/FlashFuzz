#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>  // For PyTorch C++ API

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a second tensor if we have more data
        torch::Tensor second_tensor;
        if (offset + 4 < Size) {
            second_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If not enough data for a second tensor, clone the first one
            second_tensor = input_tensor.clone();
        }
        
        // Get a scalar value from the remaining data if available
        double scalar_value = 1.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scalar_value, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Get an integer value for dimension/index operations
        int64_t dim_value = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim_value, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Test various torch operations
        try {
            // Test basic arithmetic ops
            auto result1 = torch::add(input_tensor, second_tensor);
            auto result2 = torch::sub(input_tensor, second_tensor);
            auto result3 = torch::mul(input_tensor, second_tensor);
            
            // Test scalar operations
            auto result4 = torch::add(input_tensor, scalar_value);
            auto result5 = torch::mul(input_tensor, scalar_value);
            
            // Test unary operations
            auto result6 = torch::neg(input_tensor);
            auto result7 = torch::abs(input_tensor);
            
            // Test reduction operations
            if (input_tensor.dim() > 0) {
                int64_t dim = dim_value % std::max(static_cast<int64_t>(1), static_cast<int64_t>(input_tensor.dim()));
                auto result8 = torch::sum(input_tensor, dim);
                auto result9 = torch::mean(input_tensor, dim);
            }
            
            // Test element-wise operations
            auto result10 = torch::relu(input_tensor);
            auto result11 = torch::sigmoid(input_tensor);
            
            // Test matrix operations if tensor has at least 2 dimensions
            if (input_tensor.dim() >= 2) {
                auto result12 = torch::transpose(input_tensor, 0, 1);
                
                // Test matmul if shapes allow
                if (input_tensor.size(input_tensor.dim()-1) == second_tensor.size(0)) {
                    auto result13 = torch::matmul(input_tensor, second_tensor);
                }
            }
            
            // Test tensor shape operations
            auto result14 = torch::reshape(input_tensor, second_tensor.sizes());
            
            // Test indexing operations
            if (input_tensor.dim() > 0 && input_tensor.numel() > 0) {
                int64_t index = std::abs(dim_value) % std::max(static_cast<int64_t>(1), static_cast<int64_t>(input_tensor.size(0)));
                auto result15 = torch::select(input_tensor, 0, index);
            }
            
            // Test type conversion operations
            auto result16 = input_tensor.to(torch::kFloat);
            auto result17 = input_tensor.to(torch::kInt);
            
            // Test advanced operations
            if (input_tensor.dim() > 0) {
                auto result18 = torch::softmax(input_tensor, 0);
            }
            
            // Test tensor creation ops
            auto result19 = torch::zeros_like(input_tensor);
            auto result20 = torch::ones_like(input_tensor);
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected and can be ignored
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}