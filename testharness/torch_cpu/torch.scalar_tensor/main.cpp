#include "fuzzer_utils.h"
#include <iostream>
#include <cstring>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

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
                    (void)result;
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
                    (void)result;
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
                    (void)result;
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
                    (void)result;
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
                    (void)result;
                    break;
                }
            }
        } else {
            // If we don't have enough data for a value, just use a default value
            torch::Tensor result = torch::scalar_tensor(0, scalar_type);
            (void)result;
        }
        
        // Try with TensorOptions
        if (offset < Size) {
            // Get requires_grad option (only valid for floating point types)
            bool requires_grad = (Size > offset && (Data[offset++] & 0x1));
            
            // Check if dtype supports requires_grad
            bool is_floating = (scalar_type == torch::kFloat || 
                               scalar_type == torch::kDouble ||
                               scalar_type == torch::kHalf ||
                               scalar_type == torch::kBFloat16 ||
                               scalar_type == torch::kComplexFloat ||
                               scalar_type == torch::kComplexDouble);
            
            // Only set requires_grad for floating point types
            if (!is_floating) {
                requires_grad = false;
            }
            
            // Create tensor options (CPU only since CUDA may not be available)
            auto options = torch::TensorOptions()
                .dtype(scalar_type)
                .device(torch::kCPU)
                .requires_grad(requires_grad);
            
            // Create scalar tensor with options
            double value = 0.0;
            size_t bytes_to_read = std::min(Size - offset, sizeof(double));
            if (bytes_to_read > 0) {
                std::memcpy(&value, Data + offset, bytes_to_read);
            }
            
            torch::Tensor result_with_options = torch::scalar_tensor(value, options);
            
            // Basic verification
            if (result_with_options.dim() != 0) {
                throw std::runtime_error("scalar_tensor should return 0-dim tensor");
            }
            (void)result_with_options;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}