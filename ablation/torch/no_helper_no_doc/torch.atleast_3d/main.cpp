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
        if (Size < 16) {
            return 0;
        }

        // Parse number of tensors to test (1-5)
        uint8_t num_tensors = (Data[offset] % 5) + 1;
        offset++;

        std::vector<torch::Tensor> tensors;
        
        for (uint8_t i = 0; i < num_tensors && offset < Size; i++) {
            // Parse tensor dimensions (0-6 dimensions)
            uint8_t ndim = Data[offset] % 7;
            offset++;
            
            if (offset >= Size) break;
            
            std::vector<int64_t> shape;
            
            // For 0D tensor, create scalar
            if (ndim == 0) {
                // Create scalar tensor
                float scalar_val = 0.0f;
                if (offset + sizeof(float) <= Size) {
                    std::memcpy(&scalar_val, Data + offset, sizeof(float));
                    offset += sizeof(float);
                }
                tensors.push_back(torch::tensor(scalar_val));
            }
            else {
                // Create tensor with specified dimensions
                for (uint8_t d = 0; d < ndim && offset < Size; d++) {
                    // Dimension size (1-10)
                    int64_t dim_size = (Data[offset] % 10) + 1;
                    shape.push_back(dim_size);
                    offset++;
                }
                
                if (!shape.empty()) {
                    // Parse dtype
                    uint8_t dtype_idx = Data[offset] % 6;
                    offset++;
                    
                    torch::ScalarType dtype;
                    switch (dtype_idx) {
                        case 0: dtype = torch::kFloat32; break;
                        case 1: dtype = torch::kFloat64; break;
                        case 2: dtype = torch::kInt32; break;
                        case 3: dtype = torch::kInt64; break;
                        case 4: dtype = torch::kBool; break;
                        default: dtype = torch::kFloat16; break;
                    }
                    
                    // Create tensor with random data
                    torch::Tensor tensor = torch::randn(shape, torch::TensorOptions().dtype(dtype));
                    tensors.push_back(tensor);
                }
            }
        }
        
        // Test torch::atleast_3d with each tensor
        for (const auto& tensor : tensors) {
            // Test single tensor version
            torch::Tensor result = torch::atleast_3d(tensor);
            
            // Verify result has at least 3 dimensions
            if (result.dim() < 3) {
                std::cerr << "atleast_3d failed: result has " << result.dim() << " dimensions" << std::endl;
            }
            
            // Test that the result is valid
            result.numel(); // This will throw if tensor is invalid
            
            // Test with different tensor states
            if (tensor.numel() > 0) {
                // Test with contiguous tensor
                torch::Tensor contiguous_tensor = tensor.contiguous();
                torch::Tensor contiguous_result = torch::atleast_3d(contiguous_tensor);
                contiguous_result.numel();
                
                // Test with non-contiguous tensor (if possible)
                if (tensor.dim() >= 2) {
                    torch::Tensor transposed = tensor.transpose(0, -1);
                    torch::Tensor transposed_result = torch::atleast_3d(transposed);
                    transposed_result.numel();
                }
                
                // Test with cloned tensor
                torch::Tensor cloned = tensor.clone();
                torch::Tensor cloned_result = torch::atleast_3d(cloned);
                cloned_result.numel();
            }
        }
        
        // Test with multiple tensors if we have them
        if (tensors.size() > 1) {
            // Create a vector of tensors for multi-tensor version
            std::vector<torch::Tensor> multi_results = torch::atleast_3d(tensors);
            
            // Verify all results have at least 3 dimensions
            for (size_t i = 0; i < multi_results.size(); i++) {
                if (multi_results[i].dim() < 3) {
                    std::cerr << "atleast_3d multi-tensor failed: result " << i 
                              << " has " << multi_results[i].dim() << " dimensions" << std::endl;
                }
                multi_results[i].numel(); // Verify tensor is valid
            }
        }
        
        // Test edge cases with remaining data
        if (offset < Size) {
            // Test with empty tensor
            torch::Tensor empty_tensor = torch::empty({0});
            torch::Tensor empty_result = torch::atleast_3d(empty_tensor);
            empty_result.numel();
            
            // Test with very small tensor
            torch::Tensor tiny = torch::tensor(1.0f);
            torch::Tensor tiny_result = torch::atleast_3d(tiny);
            tiny_result.numel();
            
            // Test with 1D tensor
            torch::Tensor one_d = torch::randn({5});
            torch::Tensor one_d_result = torch::atleast_3d(one_d);
            one_d_result.numel();
            
            // Test with 2D tensor
            torch::Tensor two_d = torch::randn({3, 4});
            torch::Tensor two_d_result = torch::atleast_3d(two_d);
            two_d_result.numel();
            
            // Test with already 3D tensor
            torch::Tensor three_d = torch::randn({2, 3, 4});
            torch::Tensor three_d_result = torch::atleast_3d(three_d);
            three_d_result.numel();
            
            // Test with higher dimensional tensor
            torch::Tensor high_d = torch::randn({2, 3, 4, 5});
            torch::Tensor high_d_result = torch::atleast_3d(high_d);
            high_d_result.numel();
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}