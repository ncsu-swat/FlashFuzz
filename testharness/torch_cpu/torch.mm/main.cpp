#include "fuzzer_utils.h"
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need at least 3 bytes for dimensions
        if (Size < 3) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Extract dimensions from fuzzer data
        // torch::mm requires mat1 (M x K) and mat2 (K x N)
        // Use small dimensions to keep computation manageable
        int M = (Data[offset++] % 64) + 1;  // 1-64
        int K = (Data[offset++] % 64) + 1;  // 1-64 (shared dimension)
        int N = (Data[offset++] % 64) + 1;  // 1-64
        
        // Determine dtype from remaining data
        torch::ScalarType dtype = torch::kFloat32;
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 4;
            switch (dtype_selector) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kFloat16; break;
                case 3: dtype = torch::kBFloat16; break;
            }
        }
        
        // Create properly shaped 2D tensors for matrix multiplication
        torch::Tensor mat1 = torch::randn({M, K}, torch::TensorOptions().dtype(dtype));
        torch::Tensor mat2 = torch::randn({K, N}, torch::TensorOptions().dtype(dtype));
        
        // Seed the tensors with fuzzer data if available
        if (offset < Size) {
            try {
                torch::Tensor fuzz_tensor = fuzzer_utils::createTensor(Data + offset, Size - offset, offset);
                // Use fuzz data to modify mat1 values
                if (fuzz_tensor.numel() > 0) {
                    auto flat_mat1 = mat1.flatten();
                    auto fuzz_flat = fuzz_tensor.flatten().to(dtype);
                    int copy_size = std::min(flat_mat1.numel(), fuzz_flat.numel());
                    if (copy_size > 0) {
                        flat_mat1.slice(0, 0, copy_size).copy_(fuzz_flat.slice(0, 0, copy_size));
                    }
                }
            } catch (...) {
                // Silently ignore errors in fuzzer data processing
            }
        }
        
        // Perform matrix multiplication
        torch::Tensor result = torch::mm(mat1, mat2);
        
        // Verify result properties
        if (result.defined()) {
            auto sizes = result.sizes();
            // Result should be M x N
            assert(sizes.size() == 2);
            assert(sizes[0] == M);
            assert(sizes[1] == N);
            
            // Force evaluation by computing sum
            auto sum = result.sum();
            (void)sum.item<float>();
        }
        
        // Also test the out parameter variant
        torch::Tensor out_tensor = torch::empty({M, N}, torch::TensorOptions().dtype(dtype));
        torch::mm_out(out_tensor, mat1, mat2);
        
        // Verify out tensor
        if (out_tensor.defined()) {
            auto out_sum = out_tensor.sum();
            (void)out_sum.item<float>();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}