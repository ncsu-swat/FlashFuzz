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
        
        // Parse the number of tensors to create (between 2 and 5)
        if (Size < 1) return 0;
        uint8_t num_tensors = (Data[offset++] % 4) + 2; // 2 to 5 tensors
        
        // Create a vector to store the tensors
        std::vector<torch::Tensor> tensors;
        
        // Create tensors with compatible dimensions for matrix multiplication
        for (uint8_t i = 0; i < num_tensors; ++i) {
            if (offset >= Size) break;
            
            try {
                torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Ensure tensor has at least 2 dimensions for matrix multiplication
                // If not, reshape it to a 2D tensor
                if (tensor.dim() < 2) {
                    if (tensor.dim() == 0) {
                        // Scalar tensor, reshape to 1x1
                        tensor = tensor.reshape({1, 1});
                    } else if (tensor.dim() == 1) {
                        // 1D tensor, reshape to nx1 or 1xn
                        int64_t size = tensor.size(0);
                        if (i % 2 == 0 || i == num_tensors - 1) {
                            tensor = tensor.reshape({size, 1});
                        } else {
                            tensor = tensor.reshape({1, size});
                        }
                    }
                }
                
                // For tensors after the first one, ensure matrix multiplication compatibility
                if (i > 0) {
                    // Get the last dimension of the previous tensor
                    int64_t prev_last_dim = tensors.back().size(-1);
                    
                    // Get the first dimension of the current tensor
                    int64_t curr_first_dim = tensor.size(0);
                    
                    // If dimensions don't match, reshape the current tensor
                    if (prev_last_dim != curr_first_dim) {
                        // Create a new shape with the first dimension matching the previous tensor's last dimension
                        std::vector<int64_t> new_shape;
                        new_shape.push_back(prev_last_dim);
                        
                        // Calculate the product of all other dimensions
                        int64_t remaining_elements = 1;
                        for (int64_t d = 1; d < tensor.dim(); ++d) {
                            remaining_elements *= tensor.size(d);
                        }
                        
                        // If remaining_elements is 0, set it to 1 to avoid empty tensor
                        if (remaining_elements == 0) remaining_elements = 1;
                        
                        new_shape.push_back(remaining_elements);
                        
                        // Reshape the tensor
                        tensor = tensor.reshape(new_shape);
                    }
                }
                
                tensors.push_back(tensor);
            } catch (const std::exception &e) {
                // If tensor creation fails, continue with the next tensor
                continue;
            }
        }
        
        // Need at least 2 tensors for multi_dot
        if (tensors.size() < 2) return 0;
        
        // Ensure all tensors have compatible dimensions for matrix multiplication
        for (size_t i = 0; i < tensors.size() - 1; ++i) {
            if (tensors[i].size(-1) != tensors[i+1].size(0)) {
                // If dimensions don't match, reshape the second tensor
                std::vector<int64_t> new_shape;
                new_shape.push_back(tensors[i].size(-1));
                
                // Calculate the product of all other dimensions
                int64_t remaining_elements = 1;
                for (int64_t d = 1; d < tensors[i+1].dim(); ++d) {
                    remaining_elements *= tensors[i+1].size(d);
                }
                
                // If remaining_elements is 0, set it to 1 to avoid empty tensor
                if (remaining_elements == 0) remaining_elements = 1;
                
                new_shape.push_back(remaining_elements);
                
                // Reshape the tensor
                tensors[i+1] = tensors[i+1].reshape(new_shape);
            }
        }
        
        // Apply torch.linalg.multi_dot using chain_matmul as alternative
        torch::Tensor result = torch::chain_matmul(tensors);
        
        // Optional: Try different data types
        if (Size > offset && Data[offset] % 3 == 0) {
            // Convert tensors to double and try again
            std::vector<torch::Tensor> double_tensors;
            for (const auto& t : tensors) {
                double_tensors.push_back(t.to(torch::kDouble));
            }
            torch::Tensor double_result = torch::chain_matmul(double_tensors);
        }
        
        // Optional: Try with complex tensors if available
        if (Size > offset && Data[offset] % 5 == 0) {
            try {
                std::vector<torch::Tensor> complex_tensors;
                for (const auto& t : tensors) {
                    complex_tensors.push_back(t.to(torch::kComplexFloat));
                }
                torch::Tensor complex_result = torch::chain_matmul(complex_tensors);
            } catch (const std::exception &) {
                // Ignore exceptions for complex conversion
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
