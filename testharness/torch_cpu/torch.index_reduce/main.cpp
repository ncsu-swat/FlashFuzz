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
        
        // Need at least a few bytes for basic operation
        if (Size < 8) {
            return 0;
        }
        
        // Create source tensor
        torch::Tensor src = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create index tensor (should be long type for indexing)
        torch::Tensor index;
        if (offset < Size) {
            index = fuzzer_utils::createTensor(Data, Size, offset);
            // Convert to long for indexing
            index = index.to(torch::kInt64);
        } else {
            // Create a simple index tensor if we've run out of data
            index = torch::tensor({0, 1}, torch::kInt64);
        }
        
        // Create a tensor for values to reduce
        torch::Tensor values;
        if (offset < Size) {
            values = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // Create a simple values tensor if we've run out of data
            values = torch::ones_like(src);
        }
        
        // Get reduction mode from input data
        int64_t reduce_mode = 0;
        if (offset < Size) {
            reduce_mode = static_cast<int64_t>(Data[offset++]) % 4;
        }
        
        // Map to reduction modes: "sum", "prod", "mean", "amax"
        std::string reduce;
        switch (reduce_mode) {
            case 0: reduce = "sum"; break;
            case 1: reduce = "prod"; break;
            case 2: reduce = "mean"; break;
            case 3: reduce = "amax"; break;
            default: reduce = "sum"; break;
        }
        
        // Get dimension from input data
        int64_t dim = 0;
        if (offset < Size && src.dim() > 0) {
            dim = static_cast<int64_t>(Data[offset++]) % src.dim();
        }
        
        // Get include_self flag from input data
        bool include_self = false;
        if (offset < Size) {
            include_self = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Apply index_reduce operation
        torch::Tensor result = torch::index_reduce(src, dim, index, values, reduce, include_self);
        
        // Ensure the result is valid by accessing some element
        if (result.numel() > 0) {
            auto item = result.item();
        }
        
        // Try another variant with different parameters if possible
        if (offset < Size && src.dim() > 0) {
            dim = static_cast<int64_t>(Data[offset++]) % src.dim();
            include_self = !include_self;
            
            // Try a different reduction mode
            reduce_mode = (reduce_mode + 1) % 4;
            switch (reduce_mode) {
                case 0: reduce = "sum"; break;
                case 1: reduce = "prod"; break;
                case 2: reduce = "mean"; break;
                case 3: reduce = "amax"; break;
                default: reduce = "sum"; break;
            }
            
            result = torch::index_reduce(src, dim, index, values, reduce, include_self);
            
            if (result.numel() > 0) {
                auto item = result.item();
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
