#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse dim parameter for gather
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Create index tensor
        torch::Tensor index;
        if (offset < Size) {
            index = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we don't have enough data, create a simple index tensor
            // with values in range [0, input.size(dim)-1]
            if (input.dim() > 0 && dim < input.dim() && input.size(dim) > 0) {
                std::vector<int64_t> index_shape = input.sizes().vec();
                index = torch::randint(0, std::max<int64_t>(1, input.size(dim)), index_shape);
            } else {
                // Create a simple scalar index
                index = torch::tensor(0);
            }
        }
        
        // Apply gather operation
        try {
            torch::Tensor result = torch::gather(input, dim, index);
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected and not a bug in our code
            return 0;
        }
        
        // Try sparse variant if we have enough dimensions
        if (input.dim() > 0) {
            try {
                bool sparse_grad = false;
                if (offset < Size) {
                    sparse_grad = Data[offset++] & 0x1;
                }
                torch::Tensor result_sparse = torch::gather(input, dim, index, sparse_grad);
            } catch (const c10::Error& e) {
                // PyTorch-specific exceptions are expected
                return 0;
            }
        }
        
        // Try with different dimensions if tensor has multiple dimensions
        if (input.dim() > 1) {
            int64_t alt_dim = (dim + 1) % input.dim();
            try {
                torch::Tensor result_alt_dim = torch::gather(input, alt_dim, index);
            } catch (const c10::Error& e) {
                // PyTorch-specific exceptions are expected
                return 0;
            }
        }
        
        // Try negative dimension
        if (input.dim() > 0) {
            try {
                int64_t neg_dim = -1;
                torch::Tensor result_neg_dim = torch::gather(input, neg_dim, index);
            } catch (const c10::Error& e) {
                // PyTorch-specific exceptions are expected
                return 0;
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