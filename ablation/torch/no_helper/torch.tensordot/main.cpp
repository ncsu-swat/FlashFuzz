#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic data for tensor shapes and dims specification
        if (Size < 20) return 0;

        // Generate tensor a dimensions (1-4 dims)
        int a_ndim = (Data[offset++] % 4) + 1;
        std::vector<int64_t> a_shape;
        for (int i = 0; i < a_ndim; i++) {
            int64_t dim_size = (Data[offset++] % 5) + 1; // 1-5
            a_shape.push_back(dim_size);
        }

        // Generate tensor b dimensions (1-4 dims)
        int b_ndim = (Data[offset++] % 4) + 1;
        std::vector<int64_t> b_shape;
        for (int i = 0; i < b_ndim; i++) {
            int64_t dim_size = (Data[offset++] % 5) + 1; // 1-5
            b_shape.push_back(dim_size);
        }

        if (offset >= Size) return 0;

        // Create tensors with random data
        torch::Tensor a = torch::randn(a_shape);
        torch::Tensor b = torch::randn(b_shape);

        // Test different dims specifications
        uint8_t dims_type = Data[offset++] % 4;

        if (dims_type == 0) {
            // Test with integer dims (contract last d dims of a with first d dims of b)
            int max_contractable = std::min(a_ndim, b_ndim);
            if (max_contractable > 0) {
                int dims = Data[offset++] % (max_contractable + 1); // 0 to max_contractable
                
                // For integer dims, we need matching sizes in contracted dimensions
                // Modify tensor shapes to ensure compatibility
                if (dims > 0) {
                    std::vector<int64_t> new_a_shape = a_shape;
                    std::vector<int64_t> new_b_shape = b_shape;
                    
                    // Make last 'dims' dimensions of a match first 'dims' dimensions of b
                    for (int i = 0; i < dims; i++) {
                        int64_t common_size = std::min(new_a_shape[a_ndim - dims + i], new_b_shape[i]);
                        new_a_shape[a_ndim - dims + i] = common_size;
                        new_b_shape[i] = common_size;
                    }
                    
                    a = torch::randn(new_a_shape);
                    b = torch::randn(new_b_shape);
                }
                
                torch::tensordot(a, b, dims);
            }
        }
        else if (dims_type == 1) {
            // Test with explicit dimension lists
            if (offset + 4 < Size) {
                int num_a_dims = (Data[offset++] % std::min(a_ndim, 3)) + 1;
                int num_b_dims = (Data[offset++] % std::min(b_ndim, 3)) + 1;
                
                // Make sure we contract the same number of dimensions
                int contract_dims = std::min(num_a_dims, num_b_dims);
                
                std::vector<int64_t> a_dims, b_dims;
                std::vector<int64_t> new_a_shape = a_shape;
                std::vector<int64_t> new_b_shape = b_shape;
                
                // Select dimensions to contract
                for (int i = 0; i < contract_dims && offset < Size; i++) {
                    int a_dim = Data[offset++] % a_ndim;
                    int b_dim = Data[offset++] % b_ndim;
                    
                    // Avoid duplicate dimensions
                    if (std::find(a_dims.begin(), a_dims.end(), a_dim) == a_dims.end() &&
                        std::find(b_dims.begin(), b_dims.end(), b_dim) == b_dims.end()) {
                        a_dims.push_back(a_dim);
                        b_dims.push_back(b_dim);
                        
                        // Make dimensions compatible
                        int64_t common_size = std::min(new_a_shape[a_dim], new_b_shape[b_dim]);
                        new_a_shape[a_dim] = common_size;
                        new_b_shape[b_dim] = common_size;
                    }
                }
                
                if (!a_dims.empty() && !b_dims.empty() && a_dims.size() == b_dims.size()) {
                    a = torch::randn(new_a_shape);
                    b = torch::randn(new_b_shape);
                    
                    std::vector<std::vector<int64_t>> dims_list = {a_dims, b_dims};
                    torch::tensordot(a, b, dims_list);
                }
            }
        }
        else if (dims_type == 2) {
            // Test edge cases: empty tensors, single element tensors
            if (offset < Size) {
                uint8_t edge_case = Data[offset++] % 3;
                
                if (edge_case == 0) {
                    // Empty tensor case
                    torch::Tensor empty_a = torch::empty({0, 2});
                    torch::Tensor empty_b = torch::randn({2, 3});
                    torch::tensordot(empty_a, empty_b, 1);
                }
                else if (edge_case == 1) {
                    // Single element tensors
                    torch::Tensor single_a = torch::randn({1});
                    torch::Tensor single_b = torch::randn({1});
                    torch::tensordot(single_a, single_b, 1);
                }
                else {
                    // Zero contraction dimensions
                    torch::tensordot(a, b, 0);
                }
            }
        }
        else {
            // Test with different data types and devices
            if (offset < Size) {
                uint8_t dtype_test = Data[offset++] % 4;
                
                torch::Tensor typed_a, typed_b;
                
                switch (dtype_test) {
                    case 0:
                        typed_a = a.to(torch::kFloat32);
                        typed_b = b.to(torch::kFloat32);
                        break;
                    case 1:
                        typed_a = a.to(torch::kFloat64);
                        typed_b = b.to(torch::kFloat64);
                        break;
                    case 2:
                        typed_a = a.to(torch::kInt32);
                        typed_b = b.to(torch::kInt32);
                        break;
                    default:
                        typed_a = a.to(torch::kInt64);
                        typed_b = b.to(torch::kInt64);
                        break;
                }
                
                // Simple contraction with typed tensors
                int max_contractable = std::min(typed_a.dim(), typed_b.dim());
                if (max_contractable > 0) {
                    int dims = Data[offset++] % (max_contractable + 1);
                    
                    if (dims > 0) {
                        // Ensure compatible shapes for contraction
                        auto a_shape_vec = typed_a.sizes().vec();
                        auto b_shape_vec = typed_b.sizes().vec();
                        
                        for (int i = 0; i < dims; i++) {
                            int a_idx = a_shape_vec.size() - dims + i;
                            int b_idx = i;
                            if (a_idx >= 0 && b_idx < b_shape_vec.size()) {
                                int64_t common_size = std::min(a_shape_vec[a_idx], b_shape_vec[b_idx]);
                                a_shape_vec[a_idx] = common_size;
                                b_shape_vec[b_idx] = common_size;
                            }
                        }
                        
                        typed_a = typed_a.to(torch::kFloat32).randn_like().reshape(a_shape_vec).to(typed_a.dtype());
                        typed_b = typed_b.to(torch::kFloat32).randn_like().reshape(b_shape_vec).to(typed_b.dtype());
                    }
                    
                    torch::tensordot(typed_a, typed_b, dims);
                }
            }
        }

        // Additional stress tests with larger tensors if we have enough data
        if (offset + 10 < Size) {
            // Test with broadcasting-compatible dimensions
            torch::Tensor broadcast_a = torch::randn({2, 1, 3});
            torch::Tensor broadcast_b = torch::randn({3, 4, 1});
            torch::tensordot(broadcast_a, broadcast_b, 1);
            
            // Test complex contraction patterns
            torch::Tensor complex_a = torch::randn({2, 3, 4, 5});
            torch::Tensor complex_b = torch::randn({5, 4, 3, 2});
            std::vector<std::vector<int64_t>> complex_dims = {{3, 2, 1}, {0, 1, 2}};
            torch::tensordot(complex_a, complex_b, complex_dims);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}