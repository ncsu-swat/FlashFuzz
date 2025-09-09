#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 1) {
            return 0;
        }
        
        bool mode = Data[offset] % 2;
        offset++;
        
        bool result = torch::set_flush_denormal(mode);
        
        if (offset < Size) {
            torch::Tensor test_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            torch::Tensor denormal_test = torch::tensor({1e-323}, torch::kFloat64);
            
            torch::Tensor float_test = torch::tensor({1e-40f}, torch::kFloat32);
            
            torch::Tensor combined = torch::cat({denormal_test, float_test});
            
            torch::Tensor processed = combined * 2.0;
            
            torch::set_flush_denormal(!mode);
            
            torch::Tensor another_denormal = torch::tensor({1e-324}, torch::kFloat64);
            
            torch::Tensor result_tensor = another_denormal + processed;
            
            torch::set_flush_denormal(mode);
            
            if (test_tensor.numel() > 0) {
                torch::Tensor scaled = test_tensor * 1e-300;
                torch::Tensor final_result = scaled + result_tensor.sum();
            }
        }
        
        torch::set_flush_denormal(false);
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}