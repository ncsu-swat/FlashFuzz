#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic parameters
        if (Size < 16) return 0;

        // Extract tensor dimensions (1-4D)
        int ndims = (Data[offset++] % 4) + 1;
        std::vector<int64_t> dims;
        for (int i = 0; i < ndims && offset < Size; i++) {
            int64_t dim_size = (Data[offset++] % 10) + 1; // 1-10 elements per dim
            dims.push_back(dim_size);
        }
        
        if (dims.empty()) return 0;

        // Calculate total elements
        int64_t total_elements = 1;
        for (auto d : dims) {
            total_elements *= d;
        }
        
        // Limit tensor size to prevent excessive memory usage
        if (total_elements > 1000) return 0;

        // Extract k value
        if (offset >= Size) return 0;
        int k = (Data[offset++] % static_cast<int>(total_elements)) + 1;
        
        // Extract dim parameter (optional)
        if (offset >= Size) return 0;
        bool use_dim = Data[offset++] % 2;
        int dim = 0;
        if (use_dim && offset < Size) {
            dim = Data[offset++] % ndims;
            // Handle negative dimension
            if (Data[offset++] % 2 && offset < Size) {
                dim = -dim - 1;
            }
        }

        // Extract boolean flags
        if (offset >= Size) return 0;
        bool largest = Data[offset++] % 2;
        if (offset >= Size) return 0;
        bool sorted = Data[offset++] % 2;

        // Extract dtype
        if (offset >= Size) return 0;
        int dtype_idx = Data[offset++] % 4;
        torch::ScalarType dtype;
        switch (dtype_idx) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kInt32; break;
            case 3: dtype = torch::kInt64; break;
        }

        // Create input tensor with random values
        torch::Tensor input = torch::randn(dims, torch::TensorOptions().dtype(dtype));
        
        // Fill with some deterministic values based on remaining data
        auto input_data = input.flatten();
        for (int64_t i = 0; i < input_data.size(0) && offset < Size; i++) {
            float val = static_cast<float>(Data[offset++]) / 255.0f * 20.0f - 10.0f;
            if (dtype == torch::kInt32 || dtype == torch::kInt64) {
                val = static_cast<int>(val);
            }
            input_data[i] = val;
        }

        // Test different topk call variants
        torch::Tensor values, indices;
        
        // Test 1: Basic topk with just k
        std::tie(values, indices) = torch::topk(input, k);
        
        // Test 2: topk with dim specified
        if (use_dim) {
            std::tie(values, indices) = torch::topk(input, k, dim);
        }
        
        // Test 3: topk with all parameters
        std::tie(values, indices) = torch::topk(input, k, use_dim ? dim : -1, largest, sorted);
        
        // Test 4: Edge cases
        if (k == 1) {
            // Test k=1 case
            std::tie(values, indices) = torch::topk(input, 1, use_dim ? dim : -1, largest, sorted);
        }
        
        // Test 5: Test with k equal to dimension size
        int64_t dim_size = use_dim ? input.size(dim) : input.size(-1);
        if (k <= dim_size) {
            std::tie(values, indices) = torch::topk(input, static_cast<int>(dim_size), use_dim ? dim : -1, largest, sorted);
        }
        
        // Test 6: Different data types and shapes
        if (input.numel() > 0) {
            // Test with 1D tensor
            torch::Tensor input_1d = input.flatten();
            int k_1d = std::min(k, static_cast<int>(input_1d.size(0)));
            std::tie(values, indices) = torch::topk(input_1d, k_1d, 0, largest, sorted);
            
            // Test with different k values
            for (int test_k = 1; test_k <= std::min(3, static_cast<int>(input_1d.size(0))); test_k++) {
                std::tie(values, indices) = torch::topk(input_1d, test_k, 0, largest, sorted);
            }
        }
        
        // Test 7: Test both largest=true and largest=false
        int valid_k = std::min(k, static_cast<int>(use_dim ? input.size(dim) : input.size(-1)));
        std::tie(values, indices) = torch::topk(input, valid_k, use_dim ? dim : -1, true, sorted);
        std::tie(values, indices) = torch::topk(input, valid_k, use_dim ? dim : -1, false, sorted);
        
        // Test 8: Test both sorted=true and sorted=false
        std::tie(values, indices) = torch::topk(input, valid_k, use_dim ? dim : -1, largest, true);
        std::tie(values, indices) = torch::topk(input, valid_k, use_dim ? dim : -1, largest, false);
        
        // Test 9: Test with special values if float type
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            torch::Tensor special_input = input.clone();
            if (special_input.numel() > 0) {
                special_input.flatten()[0] = std::numeric_limits<float>::infinity();
                if (special_input.numel() > 1) {
                    special_input.flatten()[1] = -std::numeric_limits<float>::infinity();
                }
                if (special_input.numel() > 2) {
                    special_input.flatten()[2] = std::numeric_limits<float>::quiet_NaN();
                }
                
                int special_k = std::min(valid_k, static_cast<int>(special_input.numel()));
                std::tie(values, indices) = torch::topk(special_input, special_k, use_dim ? dim : -1, largest, sorted);
            }
        }
        
        // Test 10: Test with zero-sized dimensions (if applicable)
        if (offset < Size && Data[offset++] % 10 == 0) {
            try {
                torch::Tensor empty_input = torch::empty({0}, torch::TensorOptions().dtype(dtype));
                if (empty_input.numel() == 0) {
                    // This should handle empty tensor case
                    std::tie(values, indices) = torch::topk(empty_input, 0);
                }
            } catch (...) {
                // Expected for invalid operations on empty tensors
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