#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t& offset, size_t size, T& out) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&out, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        if (Size < 4) return 0;
        
        size_t offset = 0;
        
        // Consume parameters for tensor creation
        uint8_t rank;
        if (!consumeBytes(Data, offset, Size, rank)) return 0;
        rank = (rank % 5) + 1; // Limit rank to 1-5
        
        // Build shape
        std::vector<int64_t> shape;
        for (int i = 0; i < rank; i++) {
            uint8_t dim;
            if (!consumeBytes(Data, offset, Size, dim)) return 0;
            // Allow 0-sized dimensions for edge cases
            shape.push_back(dim % 10);
        }
        
        // Determine dtype
        uint8_t dtype_selector;
        if (!consumeBytes(Data, offset, Size, dtype_selector)) return 0;
        
        torch::ScalarType dtype;
        switch (dtype_selector % 6) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kInt32; break;
            case 3: dtype = torch::kInt64; break;
            case 4: dtype = torch::kInt8; break;
            case 5: dtype = torch::kBool; break;
            default: dtype = torch::kFloat32;
        }
        
        // Device selection
        uint8_t device_selector;
        if (!consumeBytes(Data, offset, Size, device_selector)) return 0;
        torch::Device device = (device_selector % 2 == 0) ? torch::kCPU : torch::kCPU;
        
        // Create tensor with random data
        torch::Tensor tensor;
        if (dtype == torch::kBool) {
            tensor = torch::randint(0, 2, shape, torch::dtype(dtype).device(device)) > 0;
        } else {
            tensor = torch::randn(shape, torch::dtype(dtype).device(device));
        }
        
        // Make tensor potentially non-contiguous
        uint8_t contiguous_flag;
        if (consumeBytes(Data, offset, Size, contiguous_flag) && (contiguous_flag % 3 == 0)) {
            if (tensor.numel() > 1) {
                tensor = tensor.transpose(0, -1);
            }
        }
        
        // Consume min value
        double min_val = 0.0;
        if (offset + sizeof(float) <= Size) {
            float fval;
            consumeBytes(Data, offset, Size, fval);
            min_val = static_cast<double>(fval);
            
            // Allow special values
            uint8_t special;
            if (consumeBytes(Data, offset, Size, special)) {
                switch (special % 5) {
                    case 0: min_val = std::numeric_limits<double>::infinity(); break;
                    case 1: min_val = -std::numeric_limits<double>::infinity(); break;
                    case 2: min_val = std::numeric_limits<double>::quiet_NaN(); break;
                    case 3: min_val = std::numeric_limits<double>::max(); break;
                    case 4: min_val = std::numeric_limits<double>::lowest(); break;
                }
            }
        }
        
        // Test with scalar min value
        tensor.clamp_min_(min_val);
        
        // Also test with tensor min value if we have enough data
        if (offset + 4 <= Size) {
            uint8_t use_tensor_min;
            if (consumeBytes(Data, offset, Size, use_tensor_min) && (use_tensor_min % 2 == 0)) {
                // Create min tensor with various shapes
                uint8_t min_shape_type;
                consumeBytes(Data, offset, Size, min_shape_type);
                
                torch::Tensor min_tensor;
                switch (min_shape_type % 4) {
                    case 0: // Scalar tensor
                        min_tensor = torch::randn({}, torch::dtype(dtype).device(device));
                        break;
                    case 1: // Same shape
                        min_tensor = torch::randn(shape, torch::dtype(dtype).device(device));
                        break;
                    case 2: // Broadcastable shape
                        if (!shape.empty()) {
                            std::vector<int64_t> broadcast_shape(shape.size(), 1);
                            broadcast_shape[shape.size() - 1] = shape[shape.size() - 1];
                            min_tensor = torch::randn(broadcast_shape, torch::dtype(dtype).device(device));
                        } else {
                            min_tensor = torch::randn({1}, torch::dtype(dtype).device(device));
                        }
                        break;
                    case 3: // Different but compatible shape
                        min_tensor = torch::randn({1}, torch::dtype(dtype).device(device));
                        break;
                }
                
                // Create a new tensor for this operation
                torch::Tensor tensor2 = torch::randn(shape, torch::dtype(dtype).device(device));
                tensor2.clamp_min_(min_tensor);
            }
        }
        
        // Test edge cases with empty tensors
        if (offset < Size && Data[offset] % 10 == 0) {
            torch::Tensor empty_tensor = torch::empty({0}, torch::dtype(dtype).device(device));
            empty_tensor.clamp_min_(min_val);
            
            torch::Tensor zero_dim = torch::randn({}, torch::dtype(dtype).device(device));
            zero_dim.clamp_min_(min_val);
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}