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
        
        // Parse shape for the tensor
        std::vector<int64_t> shape = fuzzer_utils::parseShape(Data, offset, Size, rank);
        
        // Parse dtype from the next byte if available
        torch::ScalarType dtype = torch::kFloat32;
        if (offset < Size) {
            uint8_t dtype_byte = Data[offset++];
            dtype = fuzzer_utils::parseDataType(dtype_byte);
        }
        
        // Create options with the parsed dtype
        auto options = torch::TensorOptions().dtype(dtype);
        
        // Test torch::ones with different parameters
        try {
            // Basic ones with shape and options
            auto tensor1 = torch::ones(shape, options);
            
            // Test with just shape (uses default float)
            auto tensor2 = torch::ones(shape);
            
            // Test with shape and TensorOptions with dtype
            auto tensor3 = torch::ones(shape, torch::TensorOptions().dtype(dtype));
            
            // Test with IntArrayRef directly using initializer list
            if (!shape.empty() && shape[0] > 0 && shape[0] <= 100) {
                auto tensor4 = torch::ones({shape[0]});
            }
            
            // Test creating a scalar-like tensor (0-dim)
            auto tensor_scalar = torch::ones({});
            
            // Test with memory format if available (need appropriate dimensions)
            if (offset < Size) {
                uint8_t format_byte = Data[offset++];
                auto memory_format = format_byte % 4;
                
                try {
                    switch (memory_format) {
                        case 0:
                            torch::ones(shape, options.memory_format(torch::MemoryFormat::Contiguous));
                            break;
                        case 1:
                            torch::ones(shape, options.memory_format(torch::MemoryFormat::Preserve));
                            break;
                        case 2:
                            // ChannelsLast requires 4D tensor
                            if (shape.size() == 4) {
                                torch::ones(shape, options.memory_format(torch::MemoryFormat::ChannelsLast));
                            }
                            break;
                        case 3:
                            // ChannelsLast3d requires 5D tensor
                            if (shape.size() == 5) {
                                torch::ones(shape, options.memory_format(torch::MemoryFormat::ChannelsLast3d));
                            }
                            break;
                    }
                } catch (...) {
                    // Memory format may not be compatible with shape
                }
            }
            
            // Test with device if available
            if (offset < Size) {
                uint8_t device_byte = Data[offset++];
                auto device_type = device_byte % 2;
                
                try {
                    switch (device_type) {
                        case 0:
                            torch::ones(shape, options.device(torch::kCPU));
                            break;
                        case 1:
                            if (torch::cuda::is_available()) {
                                torch::ones(shape, options.device(torch::kCUDA));
                            }
                            break;
                    }
                } catch (...) {
                    // Device may not be available
                }
            }
            
            // Test with requires_grad if available (only for floating point types)
            if (offset < Size) {
                uint8_t grad_byte = Data[offset++];
                bool requires_grad = grad_byte % 2 == 1;
                
                // requires_grad only works with floating point types
                if (dtype == torch::kFloat32 || dtype == torch::kFloat64 || 
                    dtype == torch::kFloat16 || dtype == torch::kBFloat16) {
                    try {
                        torch::ones(shape, options.requires_grad(requires_grad));
                    } catch (...) {
                        // May fail for certain configurations
                    }
                }
            }
            
            // Test with pin_memory if available
            if (offset < Size) {
                uint8_t pin_byte = Data[offset++];
                bool pin_memory = pin_byte % 2 == 1;
                
                try {
                    // pinned_memory only makes sense for CPU tensors
                    torch::ones(shape, options.device(torch::kCPU).pinned_memory(pin_memory));
                } catch (...) {
                    // Pin memory may not be available
                }
            }
            
            // Test ones_like if we have a valid tensor
            if (!shape.empty()) {
                try {
                    auto base_tensor = torch::ones(shape, options);
                    auto tensor_like = torch::ones_like(base_tensor);
                } catch (...) {
                    // ones_like may fail for certain configurations
                }
            }
            
        } catch (const c10::Error& e) {
            // PyTorch specific errors are expected and should be caught silently
        } catch (const std::runtime_error& e) {
            // Runtime errors from invalid configurations
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}