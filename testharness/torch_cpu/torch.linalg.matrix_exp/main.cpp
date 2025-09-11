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
        
        // Create a tensor for matrix_exp
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // matrix_exp requires a square matrix (last two dimensions must be equal)
        // If tensor has at least 2 dimensions, we'll try to make it square
        if (input_tensor.dim() >= 2) {
            // Get the last two dimensions
            auto sizes = input_tensor.sizes();
            int64_t last_dim = sizes[sizes.size() - 1];
            int64_t second_last_dim = sizes[sizes.size() - 2];
            
            // If the last two dimensions are not equal, reshape to make them equal
            if (last_dim != second_last_dim) {
                // Choose the smaller of the two dimensions for the square matrix
                int64_t square_dim = std::min(last_dim, second_last_dim);
                
                // Create a new shape vector with the last two dimensions set to square_dim
                std::vector<int64_t> new_shape(sizes.begin(), sizes.end());
                new_shape[new_shape.size() - 1] = square_dim;
                new_shape[new_shape.size() - 2] = square_dim;
                
                // Reshape the tensor
                input_tensor = input_tensor.reshape(new_shape);
            }
        } else if (input_tensor.dim() == 1) {
            // For 1D tensor, reshape to a square matrix
            int64_t size = input_tensor.size(0);
            int64_t square_dim = static_cast<int64_t>(std::sqrt(size));
            input_tensor = input_tensor.reshape({square_dim, square_dim});
        } else if (input_tensor.dim() == 0) {
            // For scalar tensor, reshape to a 1x1 matrix
            input_tensor = input_tensor.reshape({1, 1});
        }
        
        // Convert to float or complex if needed
        if (input_tensor.dtype() == torch::kBool || 
            input_tensor.dtype() == torch::kUInt8 || 
            input_tensor.dtype() == torch::kInt8 || 
            input_tensor.dtype() == torch::kInt16 || 
            input_tensor.dtype() == torch::kInt32 || 
            input_tensor.dtype() == torch::kInt64) {
            input_tensor = input_tensor.to(torch::kFloat);
        }
        
        // Apply matrix_exp operation
        torch::Tensor result = torch::matrix_exp(input_tensor);
        
        // Optional: Check if result is finite
        if (offset < Size) {
            bool check_finite = Data[offset++] % 2 == 0;
            if (check_finite) {
                bool all_finite = torch::all(torch::isfinite(result)).item<bool>();
                if (!all_finite) {
                    // This is not an error, just an observation
                }
            }
        }
        
        // Optional: Try matrix_exp on a view of the tensor
        if (offset < Size && input_tensor.dim() >= 3) {
            uint8_t view_selector = Data[offset++];
            int64_t batch_dim = input_tensor.size(0);
            if (batch_dim > 1) {
                int64_t index = view_selector % batch_dim;
                torch::Tensor view = input_tensor.index({index});
                torch::Tensor view_result = torch::matrix_exp(view);
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
