#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Parse tensor dimensions
        auto dims = parseTensorDims(Data, Size, offset, 2, 4); // 2D to 4D tensors
        if (dims.empty()) return 0;

        // Parse data type
        auto dtype = parseDtype(Data, Size, offset);
        
        // Create input tensor
        auto input = createTensor(dims, dtype);
        if (!input.defined()) return 0;

        // Parse optional parameters
        int q = parseIntInRange(Data, Size, offset, 1, std::min(dims[dims.size()-2], dims[dims.size()-1]));
        int niter = parseIntInRange(Data, Size, offset, 1, 10);
        bool M_provided = parseBool(Data, Size, offset);

        // Test basic svd_lowrank
        auto result1 = torch::svd_lowrank(input, q);
        auto U1 = std::get<0>(result1);
        auto S1 = std::get<1>(result1);
        auto V1 = std::get<2>(result1);

        // Test with niter parameter
        auto result2 = torch::svd_lowrank(input, q, niter);
        auto U2 = std::get<0>(result2);
        auto S2 = std::get<1>(result2);
        auto V2 = std::get<2>(result2);

        // Test with M parameter if enabled
        if (M_provided && dims.size() >= 2) {
            // Create M tensor with compatible dimensions
            std::vector<int64_t> m_dims = dims;
            m_dims[m_dims.size()-1] = q; // M should have shape (..., n, q)
            
            auto M = createTensor(m_dims, dtype);
            if (M.defined()) {
                auto result3 = torch::svd_lowrank(input, q, niter, M);
                auto U3 = std::get<0>(result3);
                auto S3 = std::get<1>(result3);
                auto V3 = std::get<2>(result3);
            }
        }

        // Test edge cases
        if (dims.size() >= 2) {
            // Test with q = 1
            auto result_min = torch::svd_lowrank(input, 1);
            
            // Test with maximum possible q
            int max_q = std::min(dims[dims.size()-2], dims[dims.size()-1]);
            if (max_q > 1) {
                auto result_max = torch::svd_lowrank(input, max_q);
            }
        }

        // Test different tensor types
        if (input.dtype() != torch::kFloat32) {
            auto input_float = input.to(torch::kFloat32);
            auto result_float = torch::svd_lowrank(input_float, q);
        }

        if (input.dtype() != torch::kFloat64) {
            auto input_double = input.to(torch::kFloat64);
            auto result_double = torch::svd_lowrank(input_double, q);
        }

        // Test with complex tensors if supported
        if (input.dtype().isFloatingPoint()) {
            auto input_complex = torch::complex(input, torch::zeros_like(input));
            auto result_complex = torch::svd_lowrank(input_complex, q);
        }

        // Test with different devices if CUDA is available
        if (torch::cuda::is_available() && input.device().is_cpu()) {
            auto input_cuda = input.cuda();
            auto result_cuda = torch::svd_lowrank(input_cuda, q);
        }

        // Test reconstruction accuracy
        auto reconstructed = torch::matmul(torch::matmul(U1, torch::diag(S1)), V1.transpose(-2, -1));
        
        // Test with zero tensor
        auto zero_input = torch::zeros_like(input);
        auto zero_result = torch::svd_lowrank(zero_input, q);

        // Test with identity-like tensor (if square)
        if (dims.size() >= 2 && dims[dims.size()-2] == dims[dims.size()-1]) {
            auto eye_input = torch::eye(dims[dims.size()-1], input.options());
            if (dims.size() > 2) {
                std::vector<int64_t> batch_dims(dims.begin(), dims.end()-2);
                eye_input = eye_input.expand(dims);
            }
            auto eye_result = torch::svd_lowrank(eye_input, q);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}