#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the fuzzer data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create an uninitialized parameter
        torch::nn::UninitializedParameter param;
        
        // Try to materialize the parameter with the tensor
        param.materialize(tensor.sizes(), tensor.options());
        
        // Test parameter properties
        auto param_size = param.size();
        auto param_dtype = param.dtype();
        auto param_device = param.device();
        auto param_requires_grad = param.requires_grad();
        
        // Test parameter operations
        if (offset + 1 < Size) {
            bool requires_grad = Data[offset++] % 2 == 0;
            param.set_requires_grad(requires_grad);
        }
        
        // Test if parameter is uninitialized
        bool is_uninitialized = !param.has_uninitialized_data();
        
        // Test parameter cloning
        auto cloned_param = param.clone();
        
        // Test parameter to device
        if (torch::cuda::is_available() && offset < Size && Data[offset++] % 10 == 0) {
            param.to(torch::kCUDA);
        }
        
        // Test parameter to dtype
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            auto dtype = fuzzer_utils::parseDataType(dtype_selector);
            param.to(dtype);
        }
        
        // Test parameter zero_
        if (offset < Size && Data[offset++] % 3 == 0) {
            param.zero_();
        }
        
        // Test parameter normal_
        if (offset < Size && Data[offset++] % 3 == 0) {
            double mean = 0.0;
            double std = 1.0;
            if (offset + 1 < Size) {
                // Use fuzzer data to generate mean and std values
                mean = static_cast<double>(Data[offset++]) / 255.0 * 2.0 - 1.0;
                std = static_cast<double>(Data[offset++]) / 255.0 + 0.01;
            }
            param.normal_(mean, std);
        }
        
        // Test parameter uniform_
        if (offset < Size && Data[offset++] % 3 == 0) {
            double from = -1.0;
            double to = 1.0;
            if (offset + 1 < Size) {
                // Use fuzzer data to generate from and to values
                from = static_cast<double>(Data[offset++]) / 255.0 * 2.0 - 1.0;
                to = from + static_cast<double>(Data[offset++]) / 255.0;
            }
            param.uniform_(from, to);
        }
        
        // Test parameter fill_
        if (offset < Size && Data[offset++] % 3 == 0) {
            double value = 0.0;
            if (offset < Size) {
                value = static_cast<double>(Data[offset++]) / 255.0 * 10.0 - 5.0;
            }
            param.fill_(value);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}