#include "fuzzer_utils.h"
#include <ATen/ops/linalg_eig.h>
#include <cmath>
#include <iostream>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        if (Size < 4) {
            return 0;
        }
        
        torch::Tensor raw_input = fuzzer_utils::createTensor(Data, Size, offset);

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

        // linalg_eig requires floating point types (float, double, complex float, complex double)
        if (!input.is_floating_point() && !input.is_complex()) {
            input = input.to(torch::kFloat);
        }
        
        // Determine if we should test batched input
        bool use_batch = false;
        int64_t batch_size = 1;
        if (offset < Size) {
            use_batch = Data[offset++] & 0x1;
            if (use_batch && offset < Size) {
                batch_size = std::max<int64_t>(1, (Data[offset++] % 4) + 1);
            }
        }
        
        torch::Tensor matrix_input;
        if (use_batch && square_size > 1) {
            // Create batched input by stacking copies
            std::vector<torch::Tensor> batch_list;
            for (int64_t i = 0; i < batch_size; i++) {
                batch_list.push_back(input.clone());
            }
            matrix_input = torch::stack(batch_list, 0);
        } else {
            matrix_input = input;
        }
        
        // torch.eig is deprecated and removed; use linalg_eig instead
        // linalg_eig computes eigenvalues and eigenvectors of a square matrix
        auto result = at::linalg_eig(matrix_input);
        
        auto eigenvalues = std::get<0>(result);
        auto eigenvectors_tensor = std::get<1>(result);
        
        // Verify shapes are correct
        // eigenvalues should have shape (..., n) where n is the matrix size
        // eigenvectors should have shape (..., n, n)
        
        // Use results to prevent optimization
        auto eigenvalues_abs = torch::abs(eigenvalues);
        auto sum_eigenvalues = torch::sum(eigenvalues_abs);
        
        auto vec_abs = torch::abs(eigenvectors_tensor);
        auto vec_norm = torch::sum(vec_abs);
        
        // Volatile access to prevent dead code elimination
        volatile double ev_sum = sum_eigenvalues.item<double>();
        volatile double vec_sum = vec_norm.item<double>();
        (void)ev_sum;
        (void)vec_sum;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}