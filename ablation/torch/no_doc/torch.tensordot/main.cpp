#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }

        auto tensor_a = fuzzer_utils::createTensor(Data, Size, offset);
        if (offset >= Size) {
            return 0;
        }

        auto tensor_b = fuzzer_utils::createTensor(Data, Size, offset);
        if (offset >= Size) {
            return 0;
        }

        uint8_t dims_selector = Data[offset % Size];
        offset++;

        int dims_a_count = (dims_selector & 0x0F) % 5;
        int dims_b_count = ((dims_selector >> 4) & 0x0F) % 5;

        std::vector<int64_t> dims_a;
        std::vector<int64_t> dims_b;

        for (int i = 0; i < dims_a_count && offset < Size; i++) {
            int64_t dim = static_cast<int64_t>(Data[offset % Size]) % tensor_a.dim();
            if (dim < 0) dim = 0;
            dims_a.push_back(dim);
            offset++;
        }

        for (int i = 0; i < dims_b_count && offset < Size; i++) {
            int64_t dim = static_cast<int64_t>(Data[offset % Size]) % tensor_b.dim();
            if (dim < 0) dim = 0;
            dims_b.push_back(dim);
            offset++;
        }

        torch::tensordot(tensor_a, tensor_b, dims_a, dims_b);

        if (offset < Size) {
            int64_t single_dim_a = static_cast<int64_t>(Data[offset % Size]) % std::max(1, tensor_a.dim());
            offset++;
            if (offset < Size) {
                int64_t single_dim_b = static_cast<int64_t>(Data[offset % Size]) % std::max(1, tensor_b.dim());
                torch::tensordot(tensor_a, tensor_b, single_dim_a, single_dim_b);
            }
        }

        if (tensor_a.dim() > 0 && tensor_b.dim() > 0) {
            torch::tensordot(tensor_a, tensor_b, 1);
        }

        torch::tensordot(tensor_a, tensor_b, 0);

        if (tensor_a.dim() >= 2 && tensor_b.dim() >= 2) {
            std::vector<int64_t> last_dims_a = {tensor_a.dim() - 1};
            std::vector<int64_t> first_dims_b = {0};
            torch::tensordot(tensor_a, tensor_b, last_dims_a, first_dims_b);
        }

        if (tensor_a.dim() >= 3 && tensor_b.dim() >= 3) {
            std::vector<int64_t> multi_dims_a = {tensor_a.dim() - 2, tensor_a.dim() - 1};
            std::vector<int64_t> multi_dims_b = {0, 1};
            torch::tensordot(tensor_a, tensor_b, multi_dims_a, multi_dims_b);
        }

        std::vector<int64_t> empty_dims;
        torch::tensordot(tensor_a, tensor_b, empty_dims, empty_dims);

        if (tensor_a.dim() > 0) {
            std::vector<int64_t> all_dims_a;
            for (int i = 0; i < tensor_a.dim(); i++) {
                all_dims_a.push_back(i);
            }
            std::vector<int64_t> all_dims_b;
            for (int i = 0; i < tensor_b.dim(); i++) {
                all_dims_b.push_back(i);
            }
            if (all_dims_a.size() == all_dims_b.size()) {
                torch::tensordot(tensor_a, tensor_b, all_dims_a, all_dims_b);
            }
        }

        if (tensor_a.dim() > 0 && tensor_b.dim() > 0) {
            std::vector<int64_t> neg_dims_a = {-1};
            std::vector<int64_t> neg_dims_b = {-1};
            torch::tensordot(tensor_a, tensor_b, neg_dims_a, neg_dims_b);
        }

        if (tensor_a.dim() >= 2 && tensor_b.dim() >= 2) {
            std::vector<int64_t> dup_dims_a = {0, 0};
            std::vector<int64_t> dup_dims_b = {0, 1};
            torch::tensordot(tensor_a, tensor_b, dup_dims_a, dup_dims_b);
        }

        if (tensor_a.dim() > 0) {
            std::vector<int64_t> out_of_bounds_a = {tensor_a.dim() + 10};
            std::vector<int64_t> valid_dims_b = {0};
            if (tensor_b.dim() > 0) {
                torch::tensordot(tensor_a, tensor_b, out_of_bounds_a, valid_dims_b);
            }
        }

        std::vector<int64_t> mismatched_a = {0};
        std::vector<int64_t> mismatched_b = {0, 1};
        if (tensor_a.dim() > 0 && tensor_b.dim() >= 2) {
            torch::tensordot(tensor_a, tensor_b, mismatched_a, mismatched_b);
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}