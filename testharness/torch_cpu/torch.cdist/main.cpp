#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create two input tensors for cdist
        torch::Tensor x1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Check if we have enough data left for the second tensor
        if (offset >= Size) {
            // Try cdist with just one tensor duplicated
            torch::Tensor result = torch::cdist(x1, x1);
            return 0;
        }
        
        torch::Tensor x2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse p-norm value from remaining data
        double p = 2.0; // Default p-norm
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&p, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Parse compute_mode from remaining data
        int64_t compute_mode = 0; // Default compute_mode (use_mm_for_euclid_dist_if_necessary)
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&compute_mode, Data + offset, sizeof(int64_t));
            compute_mode = compute_mode % 3; // Limit to valid values: 0, 1, 2
            offset += sizeof(int64_t);
        }
        
        // Try different variants of cdist
        
        // Basic cdist with default parameters
        torch::Tensor result1 = torch::cdist(x1, x2);
        
        // cdist with custom p-norm
        torch::Tensor result2 = torch::cdist(x1, x2, p);
        
        // cdist with custom p-norm and compute_mode
        torch::Tensor result3 = torch::cdist(x1, x2, p, compute_mode);
        
        // Try with empty tensors or tensors with zero dimensions
        if (offset + 2 < Size) {
            std::vector<int64_t> empty_shape = {0, Data[offset] % 5 + 1};
            torch::Tensor empty_tensor = torch::empty(empty_shape);
            
            // Try cdist with an empty tensor
            try {
                torch::Tensor result_empty = torch::cdist(empty_tensor, x2);
            } catch (const std::exception&) {
                // Expected exception for incompatible shapes
            }
            
            try {
                torch::Tensor result_empty2 = torch::cdist(x1, empty_tensor);
            } catch (const std::exception&) {
                // Expected exception for incompatible shapes
            }
        }
        
        // Try with tensors that have different last dimensions
        if (offset + 4 < Size) {
            uint8_t dim1 = Data[offset++] % 5 + 1;
            uint8_t dim2 = Data[offset++] % 5 + 1;
            uint8_t dim3 = Data[offset++] % 5 + 1;
            
            torch::Tensor t1 = torch::rand({dim1, dim3});
            torch::Tensor t2 = torch::rand({dim2, dim3});
            
            // This should work as they have the same last dimension
            torch::Tensor result_diff_batch = torch::cdist(t1, t2);
            
            // Try with different last dimensions (should throw)
            try {
                torch::Tensor t3 = torch::rand({dim1, dim3 + 1});
                torch::Tensor result_incompatible = torch::cdist(t1, t3);
            } catch (const std::exception&) {
                // Expected exception for incompatible shapes
            }
        }
        
        // Try with extreme p values
        if (offset + sizeof(double) <= Size) {
            double extreme_p;
            std::memcpy(&extreme_p, Data + offset, sizeof(double));
            
            try {
                torch::Tensor result_extreme_p = torch::cdist(x1, x2, extreme_p);
            } catch (const std::exception&) {
                // May throw for invalid p values
            }
        }
        
        // Try with higher dimensional tensors
        if (offset + 2 < Size) {
            uint8_t batch_dim = Data[offset++] % 3 + 1;
            uint8_t feat_dim = Data[offset++] % 5 + 1;
            
            try {
                torch::Tensor t1 = torch::rand({batch_dim, 3, feat_dim});
                torch::Tensor t2 = torch::rand({batch_dim, 4, feat_dim});
                torch::Tensor result_batched = torch::cdist(t1, t2);
            } catch (const std::exception&) {
                // May throw for certain shapes
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