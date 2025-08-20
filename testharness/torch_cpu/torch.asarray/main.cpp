#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Try different copy options
        bool copy = (offset < Size && (Data[offset++] % 2 == 0));
        
        // Try different dtype options
        c10::optional<torch::ScalarType> dtype = c10::nullopt;
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++];
            if (dtype_selector % 3 != 0) { // 2/3 chance to specify a dtype
                dtype = fuzzer_utils::parseDataType(dtype_selector);
            }
        }
        
        // Try different device options
        torch::Device device = torch::kCPU;
        if (offset < Size) {
            uint8_t device_selector = Data[offset++];
            if (device_selector % 10 == 0) { // 1/10 chance to use CUDA if available
                if (torch::cuda::is_available()) {
                    device = torch::kCUDA;
                }
            }
        }
        
        // Apply torch::from_blob or tensor operations as asarray equivalent
        torch::Tensor result;
        
        if (!dtype.has_value()) {
            if (device == torch::kCPU) {
                result = copy ? input_tensor.clone() : input_tensor;
            } else {
                result = copy ? input_tensor.clone().to(device) : input_tensor.to(device);
            }
        } else {
            if (device == torch::kCPU) {
                result = copy ? input_tensor.clone().to(dtype.value()) : input_tensor.to(dtype.value());
            } else {
                result = copy ? input_tensor.clone().to(dtype.value()).to(device) : input_tensor.to(dtype.value()).to(device);
            }
        }
        
        // Perform some basic operations on the result to ensure it's valid
        if (result.defined()) {
            auto shape = result.sizes();
            auto numel = result.numel();
            auto dtype_result = result.dtype();
            auto device_result = result.device();
            
            // Try to access elements if tensor is not empty
            if (numel > 0) {
                if (result.is_floating_point()) {
                    auto sum = torch::sum(result).item<float>();
                } else if (result.dtype() == torch::kBool) {
                    auto any_true = torch::any(result).item<bool>();
                } else {
                    auto sum = torch::sum(result).item<int64_t>();
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