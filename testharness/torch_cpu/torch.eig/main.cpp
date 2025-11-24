#include "fuzzer_utils.h"           // General fuzzing utilities
#include <ATen/ops/linalg_eig.h>    // at::linalg_eig
#include <cmath>                    // std::sqrt
#include <iostream>                 // For cerr
#include <tuple>                    // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to create a tensor
        if (Size < 4) {
            return 0;
        }
        
        // Create a square matrix for torch.eig operation
        torch::Tensor raw_input = fuzzer_utils::createTensor(Data, Size, offset);

        // Bound total elements to keep allocations reasonable and reshape to square
        constexpr int64_t max_elements = 4096;
        auto flat = raw_input.flatten();
        if (flat.numel() == 0) {
            flat = torch::zeros({1}, raw_input.options());
        }
        int64_t limited_elems = std::min<int64_t>(flat.numel(), max_elements);
        flat = flat.narrow(0, 0, limited_elems);
        int64_t square_size = std::max<int64_t>(1, static_cast<int64_t>(std::sqrt(static_cast<double>(flat.numel()))));
        while (square_size > 1 && square_size * square_size > flat.numel()) {
            --square_size;
        }
        int64_t target_elems = square_size * square_size;
        if (target_elems == 0) {
            target_elems = 1;
        }
        torch::Tensor input = flat.narrow(0, 0, target_elems).reshape({square_size, square_size});

        // Ensure the tensor has a compatible dtype for eig (torch.eig requires floating real types)
        if (input.dtype() != torch::kFloat && input.dtype() != torch::kDouble) {
            input = input.to(torch::kFloat);
        }
        
        // Get a boolean parameter from the input data if available
        bool eigenvectors = true;
        if (offset < Size) {
            eigenvectors = Data[offset++] & 0x1;
        }
        
        // Apply eig operation (torch.eig target API)
        auto result = at::linalg_eig(input);
        
        // Access the eigenvalues and eigenvectors
        auto eigenvalues = std::get<0>(result);
        auto eigenvectors_tensor = std::get<1>(result);
        
        // Perform some operations with the results to ensure they're used
        if (eigenvectors) {
            auto vec_norm = torch::sum(torch::abs(eigenvectors_tensor));
            
            // Use the sum to prevent optimization from removing the computation
            if (vec_norm.item<double>() == -12345.6789) {
                return 1; // This will never happen, just to use the result
            }
        }
        
        // Use eigenvalues to prevent optimization from removing the computation
        auto sum_eigenvalues = torch::sum(torch::abs(eigenvalues));
        if (sum_eigenvalues.item<double>() == -12345.6789) {
            return 1; // This will never happen, just to use the result
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
