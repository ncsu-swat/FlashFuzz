#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 2 bytes for basic parameters
        if (Size < 2) {
            return 0;
        }
        
        // Parse shape for the full tensor
        uint8_t rank_byte = Data[offset++];
        uint8_t rank = fuzzer_utils::parseRank(rank_byte);
        
        // Parse shape dimensions
        std::vector<int64_t> shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
        
        // Parse fill value
        if (offset >= Size) {
            return 0;
        }
        
        // Get scalar type for the tensor
        uint8_t dtype_selector = Data[offset++];
        torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
        
        // Get a fill value from the input data
        double fill_value = 0.0;
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&fill_value, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Create options with the selected dtype
        auto options = torch::TensorOptions().dtype(dtype);
        
        // Test torch::full with various configurations
        try {
            // Basic usage with shape and fill value
            auto tensor1 = torch::full(shape, fill_value, options);
            
            // Test with scalar options
            auto tensor2 = torch::full(shape, torch::Scalar(fill_value), options);
            
            // Test with different memory layout if there's enough data
            if (offset < Size) {
                bool layout_contiguous = (Data[offset++] % 2 == 0);
                auto layout_options = options.layout(
                    layout_contiguous ? torch::kStrided : torch::kSparse);
                auto tensor3 = torch::full(shape, fill_value, layout_options);
            }
            
            // Test with different device if there's enough data
            if (offset < Size) {
                auto device_options = options.device(torch::kCPU);
                auto tensor4 = torch::full(shape, fill_value, device_options);
            }
            
            // Test with requires_grad if there's enough data
            if (offset < Size) {
                bool requires_grad = (Data[offset++] % 2 == 0);
                auto grad_options = options.requires_grad(requires_grad);
                auto tensor5 = torch::full(shape, fill_value, grad_options);
            }
            
            // Test with a different fill value type
            if (offset < Size) {
                int64_t int_fill_value = 0;
                if (offset + sizeof(int64_t) <= Size) {
                    std::memcpy(&int_fill_value, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                }
                auto tensor6 = torch::full(shape, int_fill_value, options);
            }
            
            // Test with boolean fill value
            if (offset < Size) {
                bool bool_fill_value = (Data[offset++] % 2 == 0);
                auto tensor7 = torch::full(shape, bool_fill_value, options);
            }
            
            // Test with complex fill value if dtype supports it
            if (dtype == torch::kComplexFloat || dtype == torch::kComplexDouble) {
                if (offset + sizeof(double) * 2 <= Size) {
                    double real_part, imag_part;
                    std::memcpy(&real_part, Data + offset, sizeof(double));
                    offset += sizeof(double);
                    std::memcpy(&imag_part, Data + offset, sizeof(double));
                    offset += sizeof(double);
                    
                    c10::complex<double> complex_val(real_part, imag_part);
                    auto tensor8 = torch::full(shape, complex_val, options);
                }
            }
        } catch (const c10::Error &e) {
            // PyTorch specific errors are expected and part of the test
            return 0;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}