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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create first tensor (batch1 x n x m)
        torch::Tensor input1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have at least 3 dimensions for bmm
        if (input1.dim() < 3) {
            // Expand to 3D if needed
            while (input1.dim() < 3) {
                input1 = input1.unsqueeze(0);
            }
        }
        
        // Create second tensor (batch2 x m x p)
        torch::Tensor input2;
        if (offset < Size) {
            input2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure we have at least 3 dimensions for bmm
            if (input2.dim() < 3) {
                while (input2.dim() < 3) {
                    input2 = input2.unsqueeze(0);
                }
            }
            
            // Ensure compatible dimensions for matrix multiplication
            // For bmm: input1.shape = [batch, n, m], input2.shape = [batch, m, p]
            // The batch sizes and the middle dimension must match
            
            // Get the batch size and dimensions
            int64_t batch1 = input1.size(0);
            int64_t n = input1.size(1);
            int64_t m1 = input1.size(2);
            
            int64_t batch2 = input2.size(0);
            int64_t m2 = input2.size(1);
            int64_t p = input2.size(2);
            
            // Reshape input2 to match batch size and middle dimension if needed
            if (batch1 != batch2 || m1 != m2) {
                // Create a new tensor with compatible dimensions
                auto options = torch::TensorOptions().dtype(input2.dtype());
                input2 = torch::ones({batch1, m1, p}, options);
            }
        } else {
            // If we don't have enough data for the second tensor, create a compatible one
            int64_t batch = input1.size(0);
            int64_t m = input1.size(2);
            int64_t p = 1; // Arbitrary output dimension
            
            auto options = torch::TensorOptions().dtype(input1.dtype());
            input2 = torch::ones({batch, m, p}, options);
        }
        
        // Apply bmm operation
        torch::Tensor output = torch::bmm(input1, input2);
        
        // Optional: Test edge cases with zero dimensions
        if (offset + 1 < Size) {
            uint8_t edge_case = Data[offset++];
            
            if (edge_case % 4 == 0) {
                // Test with zero batch dimension
                auto options = torch::TensorOptions().dtype(input1.dtype());
                torch::Tensor zero_batch1 = torch::ones({0, 2, 3}, options);
                torch::Tensor zero_batch2 = torch::ones({0, 3, 2}, options);
                torch::Tensor zero_output = torch::bmm(zero_batch1, zero_batch2);
            } else if (edge_case % 4 == 1) {
                // Test with zero inner dimensions
                auto options = torch::TensorOptions().dtype(input1.dtype());
                torch::Tensor zero_inner1 = torch::ones({2, 3, 0}, options);
                torch::Tensor zero_inner2 = torch::ones({2, 0, 3}, options);
                torch::Tensor zero_output = torch::bmm(zero_inner1, zero_inner2);
            } else if (edge_case % 4 == 2) {
                // Test with different data types
                auto options = torch::TensorOptions().dtype(torch::kFloat);
                torch::Tensor float_tensor1 = torch::ones({2, 3, 4}, options);
                
                options = torch::TensorOptions().dtype(torch::kDouble);
                torch::Tensor double_tensor2 = torch::ones({2, 4, 3}, options);
                
                // This will cause a type promotion
                torch::Tensor mixed_output = torch::bmm(float_tensor1, double_tensor2);
            } else {
                // Test with very large dimensions (might cause OOM but that's okay for fuzzing)
                if (offset + sizeof(int64_t) <= Size) {
                    int64_t large_dim;
                    std::memcpy(&large_dim, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    
                    // Limit to something reasonable but still potentially problematic
                    large_dim = std::abs(large_dim) % 10000 + 1;
                    
                    try {
                        auto options = torch::TensorOptions().dtype(torch::kFloat);
                        torch::Tensor large1 = torch::ones({2, large_dim, 3}, options);
                        torch::Tensor large2 = torch::ones({2, 3, large_dim}, options);
                        torch::Tensor large_output = torch::bmm(large1, large2);
                    } catch (const std::exception& e) {
                        // Expected to potentially fail with OOM
                    }
                }
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
