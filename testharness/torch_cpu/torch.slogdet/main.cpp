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
        
        // Create input tensor for slogdet
        // slogdet expects a square matrix or batch of square matrices
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // slogdet requires at least a 2D tensor (matrix)
        if (input.dim() < 2) {
            // Add dimensions if needed
            if (input.dim() == 0) {
                // Scalar to 2x2 matrix
                input = input.unsqueeze(0).unsqueeze(0).expand({2, 2});
            } else if (input.dim() == 1) {
                // Vector to square matrix
                int64_t size = input.size(0);
                if (size == 0) {
                    size = 2; // Default size if empty
                    input = torch::ones({size});
                }
                input = input.unsqueeze(0).expand({size, size});
            }
        }
        
        // Ensure the last two dimensions are square
        auto sizes = input.sizes();
        int64_t dim = input.dim();
        
        if (dim >= 2) {
            int64_t last_dim = sizes[dim-1];
            int64_t second_last_dim = sizes[dim-2];
            
            if (last_dim != second_last_dim) {
                // Make the matrix square by reshaping or expanding
                int64_t square_size = std::max(last_dim, second_last_dim);
                
                // Create a new shape with the last two dimensions being square
                std::vector<int64_t> new_shape(sizes.begin(), sizes.end());
                new_shape[dim-1] = square_size;
                new_shape[dim-2] = square_size;
                
                // Reshape or expand to make square
                if (input.numel() > 0) {
                    // Try to reshape if possible
                    if (input.numel() >= square_size * square_size) {
                        input = input.reshape(new_shape);
                    } else {
                        // Otherwise create a new tensor and copy data
                        auto new_input = torch::zeros(new_shape, input.options());
                        
                        // Copy as much data as possible
                        auto min_last = std::min(last_dim, square_size);
                        auto min_second_last = std::min(second_last_dim, square_size);
                        
                        // Create slices for copying
                        std::vector<torch::indexing::TensorIndex> src_indices(dim, torch::indexing::Slice());
                        std::vector<torch::indexing::TensorIndex> dst_indices(dim, torch::indexing::Slice());
                        
                        src_indices[dim-1] = torch::indexing::Slice(0, min_last);
                        src_indices[dim-2] = torch::indexing::Slice(0, min_second_last);
                        
                        dst_indices[dim-1] = torch::indexing::Slice(0, min_last);
                        dst_indices[dim-2] = torch::indexing::Slice(0, min_second_last);
                        
                        new_input.index_put_(dst_indices, input.index(src_indices));
                        input = new_input;
                    }
                } else {
                    // If empty tensor, create a new one with the right shape
                    input = torch::zeros(new_shape, input.options());
                }
            }
        }
        
        // Apply slogdet operation
        auto result = torch::slogdet(input);
        
        // Unpack the result (sign, logabsdet)
        auto sign = std::get<0>(result);
        auto logabsdet = std::get<1>(result);
        
        // Try to access the values to ensure computation is complete
        if (sign.numel() > 0) {
            auto sign_item = sign.item();
        }
        
        if (logabsdet.numel() > 0) {
            auto logabsdet_item = logabsdet.item();
        }
        
        // Compare results if they exist
        if (sign.numel() > 0) {
            bool signs_valid = torch::isfinite(sign).all().item<bool>();
            bool logabsdet_valid = torch::isfinite(logabsdet).all().item<bool>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
