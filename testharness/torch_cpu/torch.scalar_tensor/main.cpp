#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for scalar value type
        if (Size < 1) {
            return 0;
        }
        
        // Parse scalar value type
        uint8_t scalar_type_selector = Data[offset++];
        torch::ScalarType scalar_type = fuzzer_utils::parseDataType(scalar_type_selector);
        
        // Create a scalar value based on the remaining data
        if (offset < Size) {
            // Create different scalar types based on the selected data type
            switch (scalar_type) {
                case torch::kBool: {
                    bool value = Data[offset++] & 0x1;
                    torch::Tensor result = torch::scalar_tensor(value, scalar_type);
                    break;
                }
                case torch::kInt8:
                case torch::kUInt8:
                case torch::kInt16:
                case torch::kInt32:
                case torch::kInt64: {
                    int64_t value = 0;
                    size_t bytes_to_read = std::min(Size - offset, sizeof(int64_t));
                    if (bytes_to_read > 0) {
                        std::memcpy(&value, Data + offset, bytes_to_read);
                        offset += bytes_to_read;
                    }
                    torch::Tensor result = torch::scalar_tensor(value, scalar_type);
                    break;
                }
                case torch::kFloat:
                case torch::kDouble:
                case torch::kHalf:
                case torch::kBFloat16: {
                    double value = 0.0;
                    size_t bytes_to_read = std::min(Size - offset, sizeof(double));
                    if (bytes_to_read > 0) {
                        std::memcpy(&value, Data + offset, bytes_to_read);
                        offset += bytes_to_read;
                    }
                    torch::Tensor result = torch::scalar_tensor(value, scalar_type);
                    break;
                }
                case torch::kComplexFloat:
                case torch::kComplexDouble: {
                    double real_part = 0.0, imag_part = 0.0;
                    size_t bytes_to_read = std::min(Size - offset, sizeof(double));
                    if (bytes_to_read > 0) {
                        std::memcpy(&real_part, Data + offset, bytes_to_read);
                        offset += bytes_to_read;
                    }
                    
                    bytes_to_read = std::min(Size - offset, sizeof(double));
                    if (bytes_to_read > 0) {
                        std::memcpy(&imag_part, Data + offset, bytes_to_read);
                        offset += bytes_to_read;
                    }
                    
                    c10::complex<double> complex_value(real_part, imag_part);
                    torch::Tensor result = torch::scalar_tensor(complex_value, scalar_type);
                    break;
                }
                default: {
                    // For any other type, try with a double value
                    double value = 0.0;
                    size_t bytes_to_read = std::min(Size - offset, sizeof(double));
                    if (bytes_to_read > 0) {
                        std::memcpy(&value, Data + offset, bytes_to_read);
                        offset += bytes_to_read;
                    }
                    torch::Tensor result = torch::scalar_tensor(value, scalar_type);
                    break;
                }
            }
        } else {
            // If we don't have enough data for a value, just use a default value
            torch::Tensor result = torch::scalar_tensor(0, scalar_type);
        }
        
        // Try with options
        if (offset < Size) {
            // Get device option
            bool use_cuda = (Size > offset && (Data[offset++] & 0x1)) && torch::cuda::is_available();
            
            // Get requires_grad option
            bool requires_grad = Size > offset && (Data[offset++] & 0x1);
            
            // Create tensor options
            auto options = torch::TensorOptions()
                .dtype(scalar_type)
                .device(use_cuda ? torch::kCUDA : torch::kCPU)
                .requires_grad(requires_grad);
            
            // Create scalar tensor with options
            double value = 0.0;
            size_t bytes_to_read = std::min(Size - offset, sizeof(double));
            if (bytes_to_read > 0) {
                std::memcpy(&value, Data + offset, bytes_to_read);
            }
            
            torch::Tensor result_with_options = torch::scalar_tensor(value, options);
            
            // Verify the tensor has the expected properties
            if (result_with_options.dtype() != scalar_type ||
                result_with_options.device().type() != (use_cuda ? torch::kCUDA : torch::kCPU) ||
                result_with_options.requires_grad() != requires_grad) {
                throw std::runtime_error("Tensor properties don't match requested options");
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