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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create base tensor
        torch::Tensor base = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create exponent tensor if we have more data
        torch::Tensor exponent;
        if (offset < Size) {
            exponent = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If no more data, create a simple scalar exponent
            exponent = torch::tensor(2.0, torch::kFloat);
        }
        
        // Apply float_power in different ways to maximize coverage
        
        // 1. Basic float_power with two tensors
        torch::Tensor result1 = torch::float_power(base, exponent);
        
        // 2. Try in-place version if possible
        torch::Tensor base_copy = base.clone();
        if (base_copy.is_floating_point() || base_copy.is_complex()) {
            base_copy.float_power_(exponent);
        }
        
        // 3. Try scalar exponent version
        double scalar_exp = 0.5;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scalar_exp, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        torch::Tensor result3 = torch::float_power(base, scalar_exp);
        
        // 4. Try scalar base version
        double scalar_base = 2.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&scalar_base, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        torch::Tensor result4 = torch::float_power(scalar_base, exponent);
        
        // 6. Try with zero exponent (should return ones)
        torch::Tensor result6 = torch::float_power(base, 0.0);
        
        // 7. Try with negative exponent
        torch::Tensor result7 = torch::float_power(base, -1.0);
        
        // 8. Try with NaN/Inf values if we have floating point tensors
        if (base.is_floating_point()) {
            torch::Tensor special_values = torch::tensor({0.0, INFINITY, -INFINITY, NAN}, 
                                                        base.options());
            torch::Tensor result8 = torch::float_power(special_values, 2.0);
            torch::Tensor result9 = torch::float_power(2.0, special_values);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
