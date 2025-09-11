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

        // Need at least 3 tensors for sspaddmm: mat1, mat2, and sparse
        if (Size < 6) {
            return 0;
        }

        // Create input tensors
        torch::Tensor beta_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        if (offset >= Size) return 0;
        
        torch::Tensor alpha_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        if (offset >= Size) return 0;
        
        torch::Tensor sparse_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        if (offset >= Size) return 0;
        
        torch::Tensor mat1 = fuzzer_utils::createTensor(Data, Size, offset);
        if (offset >= Size) return 0;
        
        torch::Tensor mat2 = fuzzer_utils::createTensor(Data, Size, offset);

        // Extract scalar values for alpha and beta
        float beta = 1.0;
        float alpha = 1.0;
        
        if (beta_tensor.numel() > 0) {
            if (beta_tensor.scalar_type() == torch::ScalarType::Float) {
                beta = beta_tensor.item<float>();
            } else if (beta_tensor.scalar_type() == torch::ScalarType::Double) {
                beta = static_cast<float>(beta_tensor.item<double>());
            } else if (beta_tensor.scalar_type() == torch::ScalarType::Int) {
                beta = static_cast<float>(beta_tensor.item<int>());
            }
        }
        
        if (alpha_tensor.numel() > 0) {
            if (alpha_tensor.scalar_type() == torch::ScalarType::Float) {
                alpha = alpha_tensor.item<float>();
            } else if (alpha_tensor.scalar_type() == torch::ScalarType::Double) {
                alpha = static_cast<float>(alpha_tensor.item<double>());
            } else if (alpha_tensor.scalar_type() == torch::ScalarType::Int) {
                alpha = static_cast<float>(alpha_tensor.item<int>());
            }
        }

        // Convert sparse_tensor to sparse if it's not already
        torch::Tensor sparse;
        if (!sparse_tensor.is_sparse()) {
            // Create a sparse tensor with the same shape as sparse_tensor
            // For simplicity, we'll just convert the dense tensor to sparse
            sparse = sparse_tensor.to_sparse();
        } else {
            sparse = sparse_tensor;
        }

        // Apply sspaddmm operation
        torch::Tensor result = torch::sspaddmm(sparse, mat1, mat2, beta, alpha);

        // Try with default values for beta and alpha
        if (offset < Size) {
            torch::Tensor result3 = torch::sspaddmm(sparse, mat1, mat2);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
