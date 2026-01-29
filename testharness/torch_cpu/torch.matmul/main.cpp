#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        size_t offset = 0;
        
        // Need at least 4 bytes for minimal tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create first tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor if there's data left
        if (offset < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try to perform matmul operation
            try {
                torch::Tensor result = torch::matmul(tensor1, tensor2);
                
                // Force computation
                if (result.defined()) {
                    volatile float sum = result.to(torch::kFloat32).sum().item<float>();
                    (void)sum;
                }
            } catch (...) {
                // Shape mismatch or dtype incompatibility - expected
            }
        } else {
            // If only one tensor was created, try matmul with itself
            try {
                torch::Tensor result = torch::matmul(tensor1, tensor1);
                
                // Force computation
                if (result.defined()) {
                    volatile float sum = result.to(torch::kFloat32).sum().item<float>();
                    (void)sum;
                }
            } catch (...) {
                // Shape mismatch - expected
            }
        }
        
        // Try some edge cases if we have enough data
        if (Size > 8 && offset < Size - 4) {
            // Try vector-matrix multiplication
            try {
                std::vector<int64_t> shape1 = {3};
                torch::Tensor vec1 = torch::ones(shape1, torch::kFloat32);
                torch::Tensor result = torch::matmul(vec1, tensor1.to(torch::kFloat32));
                if (result.defined()) {
                    volatile float sum = result.sum().item<float>();
                    (void)sum;
                }
            } catch (...) {
                // Ignore exceptions from this edge case
            }
            
            // Try matmul with transposed tensor
            try {
                if (tensor1.dim() >= 2) {
                    torch::Tensor t1_float = tensor1.to(torch::kFloat32);
                    torch::Tensor transposed = t1_float.transpose(-2, -1);
                    torch::Tensor result = torch::matmul(t1_float, transposed);
                    if (result.defined()) {
                        volatile float sum = result.sum().item<float>();
                        (void)sum;
                    }
                }
            } catch (...) {
                // Ignore exceptions from this edge case
            }
            
            // Try batch matmul
            try {
                if (tensor1.dim() >= 2) {
                    int64_t m = tensor1.size(-2);
                    int64_t n = tensor1.size(-1);
                    torch::Tensor batch1 = torch::randn({2, m, n});
                    torch::Tensor batch2 = torch::randn({2, n, m});
                    torch::Tensor result = torch::matmul(batch1, batch2);
                    if (result.defined()) {
                        volatile float sum = result.sum().item<float>();
                        (void)sum;
                    }
                }
            } catch (...) {
                // Ignore exceptions from batch matmul edge case
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}