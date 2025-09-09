#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some data to work with
        if (Size < 8) {
            return 0;
        }

        // Decide whether to test single tensor or multiple tensors
        bool test_multiple = (Data[offset++] % 2) == 0;
        
        if (test_multiple) {
            // Test with multiple tensors (2-5 tensors)
            uint8_t num_tensors = 2 + (Data[offset++] % 4);
            std::vector<torch::Tensor> tensors;
            
            for (int i = 0; i < num_tensors && offset < Size; i++) {
                // Generate tensor with varying dimensions (0D to 5D)
                uint8_t num_dims = Data[offset++] % 6;
                
                if (num_dims == 0) {
                    // 0D tensor (scalar)
                    float value = generate_float_from_bytes(Data, Size, offset);
                    tensors.push_back(torch::tensor(value));
                } else {
                    // Multi-dimensional tensor
                    std::vector<int64_t> shape;
                    for (int d = 0; d < num_dims && offset < Size; d++) {
                        int64_t dim_size = 1 + (Data[offset++] % 8); // 1-8 size per dimension
                        shape.push_back(dim_size);
                    }
                    
                    if (!shape.empty()) {
                        auto tensor = generate_tensor_from_bytes(Data, Size, offset, shape);
                        tensors.push_back(tensor);
                    }
                }
            }
            
            if (!tensors.empty()) {
                // Test torch::atleast_3d with vector of tensors
                auto result = torch::atleast_3d(tensors);
                
                // Verify all output tensors have at least 3 dimensions
                for (const auto& tensor : result) {
                    if (tensor.dim() < 3) {
                        std::cerr << "Error: atleast_3d returned tensor with < 3 dimensions" << std::endl;
                    }
                }
            }
        } else {
            // Test with single tensor
            uint8_t num_dims = Data[offset++] % 6; // 0D to 5D
            
            torch::Tensor input_tensor;
            
            if (num_dims == 0) {
                // 0D tensor (scalar)
                float value = generate_float_from_bytes(Data, Size, offset);
                input_tensor = torch::tensor(value);
            } else {
                // Multi-dimensional tensor
                std::vector<int64_t> shape;
                for (int d = 0; d < num_dims && offset < Size; d++) {
                    int64_t dim_size = 1 + (Data[offset++] % 8); // 1-8 size per dimension
                    shape.push_back(dim_size);
                }
                
                if (!shape.empty()) {
                    input_tensor = generate_tensor_from_bytes(Data, Size, offset, shape);
                } else {
                    input_tensor = torch::tensor(0.0f);
                }
            }
            
            // Test torch::atleast_3d with single tensor
            auto result = torch::atleast_3d(input_tensor);
            
            // Verify output tensor has at least 3 dimensions
            if (result.dim() < 3) {
                std::cerr << "Error: atleast_3d returned tensor with < 3 dimensions" << std::endl;
            }
            
            // Test edge cases based on input dimensions
            int input_dims = input_tensor.dim();
            int output_dims = result.dim();
            
            if (input_dims == 0) {
                // 0D -> should become [1, 1, 1] shape
                if (output_dims != 3 || result.size(0) != 1 || result.size(1) != 1 || result.size(2) != 1) {
                    std::cerr << "Error: 0D tensor not properly converted to 3D" << std::endl;
                }
            } else if (input_dims == 1) {
                // 1D -> should become [1, N, 1] shape
                if (output_dims != 3 || result.size(0) != 1 || result.size(2) != 1) {
                    std::cerr << "Error: 1D tensor not properly converted to 3D" << std::endl;
                }
            } else if (input_dims == 2) {
                // 2D -> should become [M, N, 1] shape
                if (output_dims != 3 || result.size(2) != 1) {
                    std::cerr << "Error: 2D tensor not properly converted to 3D" << std::endl;
                }
            } else if (input_dims >= 3) {
                // 3D+ -> should remain unchanged
                if (output_dims != input_dims) {
                    std::cerr << "Error: 3D+ tensor dimensions changed" << std::endl;
                }
                // Check that shapes match
                for (int i = 0; i < input_dims; i++) {
                    if (input_tensor.size(i) != result.size(i)) {
                        std::cerr << "Error: 3D+ tensor shape changed" << std::endl;
                        break;
                    }
                }
            }
        }
        
        // Test with empty tensor if we have remaining data
        if (offset < Size && (Data[offset++] % 4) == 0) {
            auto empty_tensor = torch::empty({0});
            auto result = torch::atleast_3d(empty_tensor);
            if (result.dim() < 3) {
                std::cerr << "Error: atleast_3d failed on empty tensor" << std::endl;
            }
        }
        
        // Test with different dtypes if we have remaining data
        if (offset < Size) {
            uint8_t dtype_choice = Data[offset++] % 4;
            torch::Dtype dtype;
            switch (dtype_choice) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kInt32; break;
                case 3: dtype = torch::kInt64; break;
            }
            
            auto typed_tensor = torch::ones({2}, dtype);
            auto result = torch::atleast_3d(typed_tensor);
            if (result.dim() != 3 || result.dtype() != dtype) {
                std::cerr << "Error: atleast_3d failed with different dtype" << std::endl;
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}