#include "fuzzer_utils.h"
#include <iostream>

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
        
        // Need at least a few bytes for the input tensor and parameters
        if (Size < 8) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for randint_like from the remaining data
        int64_t low = 0;
        int64_t high = 10;
        
        // Parse low and high values if we have enough data
        if (offset + sizeof(int32_t) <= Size) {
            int32_t low_val;
            std::memcpy(&low_val, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            // Clamp to reasonable range to avoid overflow issues
            low = static_cast<int64_t>(low_val % 1000);
        }
        
        if (offset + sizeof(int32_t) <= Size) {
            int32_t high_val;
            std::memcpy(&high_val, Data + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
            // Clamp to reasonable range
            high = static_cast<int64_t>((high_val % 1000) + 1);
        }
        
        // Ensure high > low (required by randint_like)
        if (high <= low) {
            high = low + 1;
        }
        
        // randint_like only supports integer dtypes
        // Select from valid integer dtypes based on fuzzer data
        torch::ScalarType dtype = torch::kLong;
        if (offset < Size) {
            uint8_t dtype_selector = Data[offset++] % 4;
            switch (dtype_selector) {
                case 0: dtype = torch::kByte; break;
                case 1: dtype = torch::kInt; break;
                case 2: dtype = torch::kLong; break;
                case 3: dtype = torch::kShort; break;
            }
        }
        
        torch::Device device = torch::kCPU;
        auto options = torch::TensorOptions().dtype(dtype).device(device);
        
        // Test variant 1: randint_like(input, high)
        try {
            torch::Tensor output1 = torch::randint_like(input_tensor, high);
            // Verify shape matches input
            if (output1.sizes() != input_tensor.sizes()) {
                std::cerr << "Shape mismatch in variant 1" << std::endl;
            }
        } catch (const std::exception&) {
            // Expected for some input types
        }
        
        // Test variant 2: randint_like(input, low, high)
        try {
            torch::Tensor output2 = torch::randint_like(input_tensor, low, high);
            if (output2.sizes() != input_tensor.sizes()) {
                std::cerr << "Shape mismatch in variant 2" << std::endl;
            }
        } catch (const std::exception&) {
            // Expected for some input types
        }
        
        // Test variant 3: randint_like(input, high, options)
        try {
            torch::Tensor output3 = torch::randint_like(input_tensor, high, options);
            if (output3.sizes() != input_tensor.sizes()) {
                std::cerr << "Shape mismatch in variant 3" << std::endl;
            }
            if (output3.scalar_type() != dtype) {
                std::cerr << "Dtype mismatch in variant 3" << std::endl;
            }
        } catch (const std::exception&) {
            // Expected for some configurations
        }
        
        // Test variant 4: randint_like(input, low, high, options)
        try {
            torch::Tensor output4 = torch::randint_like(input_tensor, low, high, options);
            if (output4.sizes() != input_tensor.sizes()) {
                std::cerr << "Shape mismatch in variant 4" << std::endl;
            }
            if (output4.scalar_type() != dtype) {
                std::cerr << "Dtype mismatch in variant 4" << std::endl;
            }
        } catch (const std::exception&) {
            // Expected for some configurations
        }
        
        // Test with memory format option if we have data
        if (offset < Size) {
            try {
                auto memory_format = (Data[offset++] % 2 == 0) 
                    ? torch::MemoryFormat::Contiguous 
                    : torch::MemoryFormat::Preserve;
                auto options_with_memory = options.memory_format(memory_format);
                torch::Tensor output5 = torch::randint_like(input_tensor, low, high, options_with_memory);
            } catch (const std::exception&) {
                // Expected for some configurations
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}