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
        
        // Need at least 2 bytes for basic parameters
        if (Size < 2) {
            return 0;
        }
        
        // Parse rank from the first byte
        uint8_t rank_byte = Data[offset++];
        uint8_t rank = fuzzer_utils::parseRank(rank_byte);
        
        // Parse data type from the second byte
        uint8_t dtype_selector = Data[offset++];
        torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
        
        // Parse shape
        std::vector<int64_t> shape;
        if (offset < Size) {
            shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
        }
        
        // Ensure shape is not empty
        if (shape.empty()) {
            shape = {1};
        }
        
        // Create zeros tensor with the parsed shape and dtype
        torch::Tensor zeros_tensor = torch::zeros(shape, torch::TensorOptions().dtype(dtype));
        
        // Test additional variants of zeros
        if (Size > offset && offset + 1 < Size) {
            // Test zeros_like
            torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor zeros_like_tensor = torch::zeros_like(input_tensor);
            
            // Test zeros with options
            // Only set requires_grad for floating point types
            bool requires_grad = false;
            if (offset < Size) {
                bool want_grad = (Data[offset++] % 2 == 0);
                // requires_grad only valid for floating point types
                if (want_grad && (dtype == torch::kFloat32 || dtype == torch::kFloat64 || 
                                  dtype == torch::kFloat16 || dtype == torch::kBFloat16)) {
                    requires_grad = true;
                }
            }
            
            torch::Tensor zeros_with_options = torch::zeros(
                shape, 
                torch::TensorOptions()
                    .dtype(dtype)
                    .requires_grad(requires_grad)
            );
            
            // Test zeros with device options
            if (offset < Size) {
                offset++; // consume a byte
                torch::Tensor zeros_with_device = torch::zeros(
                    shape,
                    torch::TensorOptions()
                        .dtype(dtype)
                        .device(torch::kCPU)
                );
            }
            
            // Test zeros with memory format
            if (offset < Size) {
                uint8_t format_selector = Data[offset++];
                
                // Select memory format based on tensor dimensions
                torch::MemoryFormat memory_format = torch::MemoryFormat::Contiguous;
                
                switch (format_selector % 3) {
                    case 0: 
                        memory_format = torch::MemoryFormat::Contiguous; 
                        break;
                    case 1: 
                        // ChannelsLast requires exactly 4D tensors
                        if (shape.size() == 4) {
                            memory_format = torch::MemoryFormat::ChannelsLast;
                        }
                        break;
                    case 2: 
                        // ChannelsLast3d requires exactly 5D tensors
                        if (shape.size() == 5) {
                            memory_format = torch::MemoryFormat::ChannelsLast3d;
                        }
                        break;
                }
                
                torch::Tensor zeros_with_memory_format = torch::zeros(
                    shape,
                    torch::TensorOptions()
                        .dtype(dtype)
                        .memory_format(memory_format)
                );
            }
            
            // Test zeros with out parameter pattern
            if (offset < Size && shape.size() > 0) {
                torch::Tensor out_tensor = torch::empty(shape, torch::TensorOptions().dtype(dtype));
                torch::zeros_out(out_tensor, shape);
            }
        }
        
        // Verify that all elements are zero (only for numeric types that support comparison)
        try {
            bool all_zeros = torch::all(zeros_tensor == 0).item<bool>();
            if (!all_zeros) {
                // This should never happen - indicates a bug in PyTorch
                std::cerr << "zeros tensor contains non-zero elements" << std::endl;
            }
        } catch (...) {
            // Some dtypes may not support this comparison, that's ok
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}