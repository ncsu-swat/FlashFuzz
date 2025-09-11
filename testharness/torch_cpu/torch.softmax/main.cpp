#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        int64_t dim = 0;
        if (input.dim() > 0) {
            uint8_t dim_byte = Data[offset++];
            dim = static_cast<int64_t>(dim_byte) % (input.dim() + 1) - 1;  // Allow -1 as a dimension
        }
        
        if (offset < Size) {
            bool dim_is_name = (Data[offset++] % 2) == 1;
            
            torch::Tensor output;
            if (dim_is_name) {
                at::Dimname dimname = at::Dimname::fromSymbol(c10::Symbol::dimname("dim"));
                output = torch::softmax(input, dimname);
            } else {
                output = torch::softmax(input, dim);
            }
            
            if (offset < Size) {
                double dtype_selector = Data[offset++] % 3;
                if (dtype_selector == 0) {
                    output = torch::softmax(input, dim, torch::kFloat);
                } else if (dtype_selector == 1) {
                    output = torch::softmax(input, dim, torch::kDouble);
                } else {
                    output = torch::softmax(input, dim, torch::kHalf);
                }
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
