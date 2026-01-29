#include "fuzzer_utils.h" // General fuzzing utilities
#include <algorithm>
#include <cstring>
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
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

        // Need at least some data to proceed
        if (Size < 4)
        {
            return 0;
        }

        auto read_bounded_dim = [&](int64_t fallback) {
            int64_t raw = fallback;
            if (offset + sizeof(int64_t) <= Size)
            {
                std::memcpy(&raw, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
            }
            return std::max<int64_t>(1, std::abs(raw) % 32);
        };

        const int64_t sparse_dim_m = read_bounded_dim(4);
        const int64_t sparse_dim_k = read_bounded_dim(4);
        const int64_t dense_dim_n = read_bounded_dim(4);

        int64_t nnz_hint = 0;
        if (offset + sizeof(int64_t) <= Size)
        {
            std::memcpy(&nnz_hint, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        const int64_t max_nnz = std::min<int64_t>(sparse_dim_m * sparse_dim_k, 512);
        const int64_t nnz = std::min<int64_t>(max_nnz, std::max<int64_t>(0, std::abs(nnz_hint) % (max_nnz + 1)));

        torch::Tensor indices = torch::zeros({2, nnz}, torch::kLong);
        if (nnz > 0)
        {
            torch::Tensor idx_source = fuzzer_utils::createTensor(Data, Size, offset).to(torch::kLong).view(-1);
            if (idx_source.numel() < nnz * 2)
            {
                idx_source = torch::cat({idx_source, torch::zeros({nnz * 2 - idx_source.numel()}, torch::kLong)});
            }
            idx_source = idx_source.narrow(0, 0, nnz * 2).reshape({2, nnz});
            auto size_tensor = torch::tensor({sparse_dim_m, sparse_dim_k}, torch::kLong).unsqueeze(1);
            indices = torch::remainder(idx_source.abs(), size_tensor);
        }

        auto value_options = torch::TensorOptions().dtype(torch::kFloat);
        torch::Tensor values = torch::zeros({nnz}, value_options);
        if (nnz > 0)
        {
            try
            {
                auto raw_values = fuzzer_utils::createTensor(Data, Size, offset).to(torch::kFloat).flatten();
                if (raw_values.numel() >= nnz)
                {
                    values = raw_values.narrow(0, 0, nnz).clone();
                }
                else
                {
                    values = torch::cat({raw_values, torch::zeros({nnz - raw_values.numel()}, value_options)});
                }
            }
            catch (...)
            {
                values = torch::ones({nnz}, value_options);
            }
        }

        auto sparse = torch::sparse_coo_tensor(indices, values, {sparse_dim_m, sparse_dim_k}).coalesce();

        torch::Tensor dense;
        try
        {
            dense = fuzzer_utils::createTensor(Data, Size, offset).to(torch::kFloat);
        }
        catch (...)
        {
            dense = torch::randn({sparse_dim_k, dense_dim_n}, value_options);
        }

        if (dense.dim() != 2 || dense.size(0) != sparse_dim_k)
        {
            dense = dense.flatten();
            const int64_t required = sparse_dim_k * dense_dim_n;
            if (dense.numel() < required)
            {
                dense = torch::cat({dense, torch::zeros({required - dense.numel()}, value_options)});
            }
            dense = dense.narrow(0, 0, required).reshape({sparse_dim_k, dense_dim_n});
        }
        else if (dense.size(1) != dense_dim_n)
        {
            const int64_t cols = std::min<int64_t>(dense.size(1), dense_dim_n);
            dense = dense.narrow(1, 0, cols);
            if (cols < dense_dim_n)
            {
                dense = torch::cat({dense, torch::zeros({sparse_dim_k, dense_dim_n - cols}, value_options)}, 1);
            }
        }

        // torch.dsmm in Python is equivalent to torch::mm with sparse @ dense in C++
        torch::Tensor result = torch::mm(sparse, dense);
        (void)result.sum().item<double>();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}