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
        
        // Create a tensor from the input data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Try different ways to call torch::tensor
        try {
            // Try with options
            torch::TensorOptions options;
            
            // Use remaining bytes to determine options if available
            if (offset + 1 < Size) {
                uint8_t dtype_selector = Data[offset++];
                options = options.dtype(fuzzer_utils::parseDataType(dtype_selector));
            }
            
            if (offset + 1 < Size) {
                uint8_t requires_grad = Data[offset++] % 2;
                options = options.requires_grad(requires_grad == 1);
            }
            
            // Try with scalar values
            if (input_tensor.numel() > 0) {
                // Get a scalar from the tensor if possible
                try {
                    torch::Scalar scalar = input_tensor.item();
                    torch::Tensor scalar_tensor = torch::tensor(scalar.toDouble(), options);
                } catch (const std::exception&) {
                    // If item() fails, try with a simple scalar
                    torch::Tensor scalar_tensor = torch::tensor(3.14, options);
                }
            }
            
            // Try with vector/list input
            if (input_tensor.dim() == 1 && input_tensor.numel() > 0) {
                std::vector<float> vec_data;
                try {
                    // Convert tensor to vector if it's a 1D tensor
                    auto accessor = input_tensor.accessor<float, 1>();
                    for (int i = 0; i < accessor.size(0); i++) {
                        vec_data.push_back(accessor[i]);
                    }
                    torch::Tensor vec_tensor = torch::tensor(vec_data, options);
                } catch (const std::exception&) {
                    // Fallback to a simple vector
                    std::vector<int> simple_vec = {1, 2, 3};
                    torch::Tensor vec_tensor = torch::tensor(simple_vec, options);
                }
            }
            
            // Try with nested vectors for multi-dimensional tensors
            if (offset + 1 < Size) {
                torch::Tensor nested_tensor = torch::tensor({{1, 2}, {3, 4}}, options);
            }
            
            // Try with empty list
            std::vector<int> empty_vec;
            torch::Tensor empty_tensor = torch::tensor(empty_vec, options);
            
            // Try with boolean values
            if (offset + 1 < Size) {
                bool bool_val = (Data[offset++] % 2) == 1;
                torch::Tensor bool_tensor = torch::tensor(bool_val, options);
            }
        } catch (const std::exception& e) {
            // Catch exceptions from torch::tensor operations but continue fuzzing
        }
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
