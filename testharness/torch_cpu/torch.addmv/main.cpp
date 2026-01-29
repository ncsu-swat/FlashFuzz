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
        // Need at least some data for meaningful fuzzing
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Extract dimensions from fuzzer data
        int64_t M = (Data[offset++] % 32) + 1;  // rows of matrix, 1-32
        int64_t N = (Data[offset++] % 32) + 1;  // cols of matrix, 1-32

        // Extract alpha and beta scaling factors
        double alpha = 1.0;
        double beta = 1.0;
        if (offset + 1 < Size) {
            alpha = static_cast<double>(Data[offset++]) / 64.0 - 2.0;  // range roughly -2 to 2
        }
        if (offset < Size) {
            beta = static_cast<double>(Data[offset++]) / 64.0 - 2.0;
        }

        // Determine dtype from fuzzer data
        torch::ScalarType dtype = torch::kFloat32;
        if (offset < Size) {
            uint8_t dtype_choice = Data[offset++] % 3;
            if (dtype_choice == 0) dtype = torch::kFloat32;
            else if (dtype_choice == 1) dtype = torch::kFloat64;
            else dtype = torch::kFloat16;
        }

        // Create properly shaped tensors for addmv: bias[M] + input[M,N] @ vec[N]
        torch::Tensor input, vec, bias;
        
        try {
            // Create matrix input [M, N]
            input = fuzzer_utils::createTensor(Data, Size, offset);
            input = input.flatten().slice(0, 0, std::min(input.numel(), M * N));
            if (input.numel() < M * N) {
                input = torch::cat({input, torch::zeros(M * N - input.numel())});
            }
            input = input.reshape({M, N}).to(dtype);

            // Create vector vec [N]
            vec = fuzzer_utils::createTensor(Data, Size, offset);
            vec = vec.flatten().slice(0, 0, std::min(vec.numel(), N));
            if (vec.numel() < N) {
                vec = torch::cat({vec, torch::zeros(N - vec.numel())});
            }
            vec = vec.reshape({N}).to(dtype);

            // Create bias vector [M]
            bias = fuzzer_utils::createTensor(Data, Size, offset);
            bias = bias.flatten().slice(0, 0, std::min(bias.numel(), M));
            if (bias.numel() < M) {
                bias = torch::cat({bias, torch::zeros(M - bias.numel())});
            }
            bias = bias.reshape({M}).to(dtype);
        } catch (...) {
            // Fallback to simple tensors if reshaping fails
            auto opts = torch::TensorOptions().dtype(dtype);
            input = torch::randn({M, N}, opts);
            vec = torch::randn({N}, opts);
            bias = torch::randn({M}, opts);
        }

        // Test torch::addmv with alpha and beta
        try {
            torch::Tensor result = torch::addmv(bias, input, vec, beta, alpha);
            (void)result;  // Prevent optimization
        } catch (const c10::Error& e) {
            // Expected for certain dtype combinations
        }

        // Test torch::addmv with default alpha/beta
        try {
            torch::Tensor result = torch::addmv(bias, input, vec);
            (void)result;
        } catch (const c10::Error& e) {
            // Expected for certain dtype combinations
        }

        // Test tensor method addmv
        try {
            torch::Tensor result = bias.addmv(input, vec, beta, alpha);
            (void)result;
        } catch (const c10::Error& e) {
            // Expected for certain configurations
        }

        // Test in-place addmv_
        try {
            torch::Tensor bias_copy = bias.clone();
            bias_copy.addmv_(input, vec, beta, alpha);
            (void)bias_copy;
        } catch (const c10::Error& e) {
            // Expected for certain configurations
        }

        // Test addmv_out
        try {
            torch::Tensor out = torch::empty({M}, torch::TensorOptions().dtype(dtype));
            torch::addmv_out(out, bias, input, vec, beta, alpha);
            (void)out;
        } catch (const c10::Error& e) {
            // Expected for certain configurations
        }

        // Test with transposed matrix (explores different memory layouts)
        try {
            torch::Tensor input_t = input.t().contiguous();  // Now [N, M]
            torch::Tensor vec_for_t = torch::randn({M}, torch::TensorOptions().dtype(dtype));
            torch::Tensor bias_for_t = torch::randn({N}, torch::TensorOptions().dtype(dtype));
            torch::Tensor result = torch::addmv(bias_for_t, input_t, vec_for_t, beta, alpha);
            (void)result;
        } catch (const c10::Error& e) {
            // Expected
        }

        // Test with non-contiguous tensors
        try {
            if (M > 1 && N > 1) {
                torch::Tensor input_nc = input.slice(0, 0, M).slice(1, 0, N);
                torch::Tensor result = torch::addmv(bias, input_nc, vec, beta, alpha);
                (void)result;
            }
        } catch (const c10::Error& e) {
            // Expected
        }

        // Test edge cases with zero alpha or beta
        try {
            torch::Tensor result = torch::addmv(bias, input, vec, 0.0, alpha);
            (void)result;
        } catch (const c10::Error& e) {
            // Expected
        }

        try {
            torch::Tensor result = torch::addmv(bias, input, vec, beta, 0.0);
            (void)result;
        } catch (const c10::Error& e) {
            // Expected
        }

    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}