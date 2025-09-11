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
        
        // Need at least 2 bytes for n and m parameters
        if (Size < 2) {
            return 0;
        }
        
        // Extract n (number of rows)
        int64_t n = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&n, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        } else {
            n = static_cast<int64_t>(Data[offset++]);
        }
        
        // Extract m (number of columns, optional)
        int64_t m = 0;
        bool use_m = false;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&m, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            use_m = true;
        } else if (offset < Size) {
            m = static_cast<int64_t>(Data[offset++]);
            use_m = true;
        }
        
        // Extract dtype (optional)
        torch::ScalarType dtype = torch::kFloat;
        bool use_dtype = false;
        if (offset < Size) {
            dtype = fuzzer_utils::parseDataType(Data[offset++]);
            use_dtype = true;
        }
        
        // Extract device (optional)
        torch::Device device = torch::kCPU;
        
        // Create identity matrix with different parameter combinations
        torch::Tensor result;
        
        // Test different combinations of parameters
        if (!use_m && !use_dtype) {
            // eye(n)
            result = torch::eye(n);
        } else if (use_m && !use_dtype) {
            // eye(n, m)
            result = torch::eye(n, m);
        } else if (!use_m && use_dtype) {
            // eye(n, dtype=dtype)
            result = torch::eye(n, torch::TensorOptions().dtype(dtype).device(device));
        } else {
            // eye(n, m, dtype=dtype)
            result = torch::eye(n, m, torch::TensorOptions().dtype(dtype).device(device));
        }
        
        // Perform some operations on the result to ensure it's used
        if (result.numel() > 0) {
            auto sum = result.sum();
            auto trace = result.trace();
            
            // Test diagonal extraction
            if (result.dim() >= 2) {
                auto diag = result.diag();
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
