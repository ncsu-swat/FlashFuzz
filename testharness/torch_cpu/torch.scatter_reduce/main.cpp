#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for basic operation
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create index tensor with appropriate dtype (must be long)
        torch::Tensor index;
        if (offset < Size) {
            index = fuzzer_utils::createTensor(Data, Size, offset).to(torch::kLong);
        } else {
            // If we've consumed all data, create a simple index tensor
            index = torch::tensor({0}, torch::kLong);
        }
        
        // Create src tensor
        torch::Tensor src;
        if (offset < Size) {
            src = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we've consumed all data, create a simple src tensor
            src = torch::ones_like(input);
        }
        
        // Get dimension to scatter along
        int64_t dim = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // If input is not a scalar, ensure dim is valid
            if (input.dim() > 0) {
                dim = dim % input.dim();
                if (dim < 0) {
                    dim += input.dim();
                }
            }
        }
        
        // Get reduce operation
        std::string reduce = "sum";
        if (offset < Size) {
            uint8_t reduce_op = Data[offset++];
            switch (reduce_op % 5) {
                case 0: reduce = "sum"; break;
                case 1: reduce = "prod"; break;
                case 2: reduce = "mean"; break;
                case 3: reduce = "amax"; break;
                case 4: reduce = "amin"; break;
            }
        }
        
        // Get include_self flag
        bool include_self = true;
        if (offset < Size) {
            include_self = static_cast<bool>(Data[offset++] & 0x01);
        }
        
        // Try different combinations of scatter_reduce
        try {
            // Basic scatter_reduce
            torch::Tensor result1 = torch::scatter_reduce(input, dim, index, src, reduce, include_self);
            
            // Try in-place version if possible
            torch::Tensor input_copy = input.clone();
            input_copy.scatter_reduce_(dim, index, src, reduce, include_self);
            
            // Try with different input types if possible
            if (input.scalar_type() != torch::kBool && 
                input.scalar_type() != torch::kBFloat16 && 
                input.scalar_type() != torch::kHalf) {
                torch::Tensor input_float = input.to(torch::kFloat);
                torch::Tensor src_float = src.to(torch::kFloat);
                torch::Tensor result3 = torch::scatter_reduce(
                    input_float, dim, index, src_float, reduce, include_self);
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific exceptions are expected and part of testing
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}