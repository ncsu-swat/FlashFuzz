#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Call cudnn_is_acceptable on the tensor
        volatile bool is_acceptable = torch::cudnn_is_acceptable(tensor);
        (void)is_acceptable;
        
        // Try with different tensor types and shapes if we have more data
        if (offset + 2 < Size) {
            torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            volatile bool is_acceptable2 = torch::cudnn_is_acceptable(tensor2);
            (void)is_acceptable2;
        }
        
        // Try with a tensor that has special properties
        if (offset + 2 < Size) {
            // Create a tensor with potentially challenging properties
            uint8_t dtype_selector = Data[offset++] % 12;
            torch::ScalarType dtype;
            
            // Select different dtypes to test
            switch (dtype_selector) {
                case 0: dtype = torch::kFloat; break;
                case 1: dtype = torch::kDouble; break;
                case 2: dtype = torch::kHalf; break;
                case 3: dtype = torch::kBFloat16; break;
                case 4: dtype = torch::kInt8; break;
                case 5: dtype = torch::kInt16; break;
                case 6: dtype = torch::kInt32; break;
                case 7: dtype = torch::kInt64; break;
                case 8: dtype = torch::kUInt8; break;
                case 9: dtype = torch::kBool; break;
                case 10: dtype = torch::kComplexFloat; break;
                case 11: dtype = torch::kComplexDouble; break;
                default: dtype = torch::kFloat;
            }
            
            // Create tensors with various shapes
            std::vector<int64_t> shape;
            uint8_t rank = (offset < Size) ? (Data[offset++] % 5) + 1 : 2;
            
            for (uint8_t i = 0; i < rank && offset < Size; i++) {
                int64_t dim = static_cast<int64_t>(Data[offset++] % 64) + 1;
                shape.push_back(dim);
            }
            
            // Ensure we have at least one dimension
            if (shape.empty()) {
                shape.push_back(1);
            }
            
            // Create tensor with the specified properties
            torch::Tensor special_tensor = torch::empty(shape, torch::TensorOptions().dtype(dtype));
            
            // Test cudnn_is_acceptable with this tensor
            volatile bool is_special_acceptable = torch::cudnn_is_acceptable(special_tensor);
            (void)is_special_acceptable;
            
            // Try with non-contiguous tensor if possible
            if (shape.size() >= 2 && shape[0] > 1 && shape[1] > 1) {
                torch::Tensor non_contiguous = special_tensor.transpose(0, 1);
                volatile bool is_non_contiguous_acceptable = torch::cudnn_is_acceptable(non_contiguous);
                (void)is_non_contiguous_acceptable;
            }
        }
        
        // Try with empty tensor
        torch::Tensor empty_tensor = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat));
        volatile bool is_empty_acceptable = torch::cudnn_is_acceptable(empty_tensor);
        (void)is_empty_acceptable;
        
        // Try with scalar tensor
        torch::Tensor scalar_tensor = torch::tensor(1.0f);
        volatile bool is_scalar_acceptable = torch::cudnn_is_acceptable(scalar_tensor);
        (void)is_scalar_acceptable;
        
        // Try with various memory formats
        if (offset + 4 < Size) {
            int64_t n = (Data[offset++] % 4) + 1;
            int64_t c = (Data[offset++] % 8) + 1;
            int64_t h = (Data[offset++] % 8) + 1;
            int64_t w = (Data[offset++] % 8) + 1;
            
            // Contiguous NCHW tensor
            torch::Tensor nchw_tensor = torch::empty({n, c, h, w}, torch::TensorOptions().dtype(torch::kFloat));
            volatile bool is_nchw_acceptable = torch::cudnn_is_acceptable(nchw_tensor);
            (void)is_nchw_acceptable;
            
            // Try channels_last format if available
            try {
                torch::Tensor channels_last_tensor = nchw_tensor.contiguous(torch::MemoryFormat::ChannelsLast);
                volatile bool is_channels_last_acceptable = torch::cudnn_is_acceptable(channels_last_tensor);
                (void)is_channels_last_acceptable;
            } catch (...) {
                // Silently ignore if channels_last is not supported for this shape
            }
        }
        
        // Try with strided tensor
        if (offset < Size) {
            int64_t size_val = (Data[offset++] % 16) + 2;
            torch::Tensor base_tensor = torch::empty({size_val * 2}, torch::TensorOptions().dtype(torch::kFloat));
            // Create a strided view
            torch::Tensor strided_tensor = base_tensor.slice(0, 0, size_val * 2, 2);
            volatile bool is_strided_acceptable = torch::cudnn_is_acceptable(strided_tensor);
            (void)is_strided_acceptable;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}