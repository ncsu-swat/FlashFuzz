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
        
        // Skip if not enough data
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test various return_types operations
        
        // 1. Test torch::max with dim
        if (tensor.dim() > 0) {
            try {
                int64_t dim = 0;
                auto max_result = torch::max(tensor, dim);
                torch::Tensor values = std::get<0>(max_result);
                torch::Tensor indices = std::get<1>(max_result);
            } catch (const std::exception&) {
                // max with dim may fail for certain inputs
            }
        }
        
        // 2. Test torch::min with dim
        if (tensor.dim() > 0) {
            try {
                int64_t dim = 0;
                auto min_result = torch::min(tensor, dim);
                torch::Tensor values = std::get<0>(min_result);
                torch::Tensor indices = std::get<1>(min_result);
            } catch (const std::exception&) {
                // min with dim may fail for certain inputs
            }
        }
        
        // 3. Test torch::sort
        if (tensor.dim() > 0) {
            try {
                int64_t dim = 0;
                bool descending = (offset < Size && Data[offset++] % 2 == 0);
                auto sort_result = torch::sort(tensor, dim, descending);
                torch::Tensor values = std::get<0>(sort_result);
                torch::Tensor indices = std::get<1>(sort_result);
            } catch (const std::exception&) {
                // sort may fail for certain inputs
            }
        }
        
        // 4. Test torch::topk
        if (tensor.dim() > 0 && tensor.size(0) > 0) {
            try {
                int64_t k = 1;
                if (tensor.size(0) > 1) {
                    k = (offset < Size) ? (Data[offset++] % tensor.size(0)) + 1 : 1;
                }
                int64_t dim = 0;
                bool largest = (offset < Size && Data[offset++] % 2 == 0);
                bool sorted = (offset < Size && Data[offset++] % 2 == 0);
                
                auto topk_result = torch::topk(tensor, k, dim, largest, sorted);
                torch::Tensor values = std::get<0>(topk_result);
                torch::Tensor indices = std::get<1>(topk_result);
            } catch (const std::exception&) {
                // topk may fail for certain inputs
            }
        }
        
        // 5. Test torch::svd
        try {
            auto svd_result = torch::svd(tensor);
            torch::Tensor U = std::get<0>(svd_result);
            torch::Tensor S = std::get<1>(svd_result);
            torch::Tensor V = std::get<2>(svd_result);
        } catch (const std::exception&) {
            // SVD may fail for certain inputs
        }
        
        // 6. Test torch::mode
        if (tensor.dim() > 0) {
            try {
                int64_t dim = 0;
                auto mode_result = torch::mode(tensor, dim);
                torch::Tensor values = std::get<0>(mode_result);
                torch::Tensor indices = std::get<1>(mode_result);
            } catch (const std::exception&) {
                // mode may fail for certain inputs
            }
        }
        
        // 7. Test torch::median with dim
        if (tensor.dim() > 0) {
            try {
                int64_t dim = 0;
                auto median_result = torch::median(tensor, dim);
                torch::Tensor values = std::get<0>(median_result);
                torch::Tensor indices = std::get<1>(median_result);
            } catch (const std::exception&) {
                // median may fail for certain inputs
            }
        }
        
        // 8. Test torch::kthvalue
        if (tensor.dim() > 0 && tensor.size(0) > 0) {
            try {
                int64_t k = 1;
                if (tensor.size(0) > 1) {
                    k = (offset < Size) ? (Data[offset++] % tensor.size(0)) + 1 : 1;
                }
                int64_t dim = 0;
                bool keepdim = (offset < Size && Data[offset++] % 2 == 0);
                
                auto kthvalue_result = torch::kthvalue(tensor, k, dim, keepdim);
                torch::Tensor values = std::get<0>(kthvalue_result);
                torch::Tensor indices = std::get<1>(kthvalue_result);
            } catch (const std::exception&) {
                // kthvalue may fail for certain inputs
            }
        }
        
        // 9. Test torch::qr
        try {
            auto qr_result = torch::qr(tensor);
            torch::Tensor Q = std::get<0>(qr_result);
            torch::Tensor R = std::get<1>(qr_result);
        } catch (const std::exception&) {
            // QR decomposition may fail for certain inputs
        }
        
        // 10. Test torch::lu
        try {
            auto lu_result = torch::lu(tensor);
            torch::Tensor LU = std::get<0>(lu_result);
            torch::Tensor pivots = std::get<1>(lu_result);
        } catch (const std::exception&) {
            // LU factorization may fail for non-square matrices or other reasons
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
