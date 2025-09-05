#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

// Helper to consume bytes from the fuzzer input
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
        
        // Consume configuration bytes
        uint8_t rank;
        if (!consumeBytes(Data, offset, Size, rank)) return 0;
        rank = (rank % 5) + 1; // Rank between 1 and 5
        
        uint8_t dtype_choice;
        if (!consumeBytes(Data, offset, Size, dtype_choice)) return 0;
        
        uint8_t device_choice;
        if (!consumeBytes(Data, offset, Size, device_choice)) return 0;
        
        uint8_t requires_grad;
        if (!consumeBytes(Data, offset, Size, requires_grad)) return 0;
        
        // Build shape
        std::vector<int64_t> shape;
        for (int i = 0; i < rank; i++) {
            uint8_t dim;
            if (!consumeBytes(Data, offset, Size, dim)) return 0;
            shape.push_back(dim % 10); // Keep dimensions small, including 0
        }
        
        // Select dtype
        torch::ScalarType dtype;
        switch (dtype_choice % 6) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kInt32; break;
            case 3: dtype = torch::kInt64; break;
            case 4: dtype = torch::kInt8; break;
            case 5: dtype = torch::kBool; break;
            default: dtype = torch::kFloat32;
        }
        
        // Select device
        torch::Device device = (device_choice % 2 == 0) ? torch::kCPU : torch::kCPU;
        // Note: Using CPU only to avoid CUDA availability issues
        
        // Create options
        auto options = torch::TensorOptions()
            .dtype(dtype)
            .device(device)
            .requires_grad((requires_grad % 2) == 1 && (dtype == torch::kFloat32 || dtype == torch::kFloat64));
        
        // Create tensor with random data
        torch::Tensor tensor;
        int64_t numel = 1;
        for (auto d : shape) numel *= d;
        
        if (numel == 0) {
            tensor = torch::empty(shape, options);
        } else if (numel > 0 && numel <= 1000000) {
            // Fill tensor with fuzzer data
            if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
                std::vector<float> values;
                for (int64_t i = 0; i < numel; i++) {
                    float val;
                    if (!consumeBytes(Data, offset, Size, val)) {
                        val = static_cast<float>(i);
                    }
                    values.push_back(val);
                }
                tensor = torch::from_blob(values.data(), shape, torch::kFloat32).to(options);
            } else if (dtype == torch::kBool) {
                tensor = torch::randint(0, 2, shape, options.dtype(torch::kInt8)).to(torch::kBool);
            } else {
                tensor = torch::randint(-100, 100, shape, options);
            }
        } else {
            return 0; // Skip very large tensors
        }
        
        // Get min value for clamp_min
        float min_val;
        if (!consumeBytes(Data, offset, Size, min_val)) {
            min_val = 0.0f;
        }
        
        // Test different min value types
        uint8_t min_type;
        if (!consumeBytes(Data, offset, Size, min_type)) {
            min_type = 0;
        }
        
        // Apply clamp_min with different argument types
        torch::Tensor result;
        switch (min_type % 3) {
            case 0:
                // Scalar min
                result = torch::clamp_min(tensor, min_val);
                break;
            case 1:
                // Tensor min (same shape)
                if (numel > 0) {
                    auto min_tensor = torch::full(shape, min_val, options);
                    result = torch::clamp_min(tensor, min_tensor);
                } else {
                    result = torch::clamp_min(tensor, min_val);
                }
                break;
            case 2:
                // Tensor min (broadcastable shape)
                if (rank > 0 && shape[rank-1] > 0) {
                    std::vector<int64_t> broadcast_shape(rank, 1);
                    broadcast_shape[rank-1] = shape[rank-1];
                    auto min_tensor = torch::full(broadcast_shape, min_val, options);
                    result = torch::clamp_min(tensor, min_tensor);
                } else {
                    result = torch::clamp_min(tensor, min_val);
                }
                break;
        }
        
        // Test in-place operation
        uint8_t inplace;
        if (consumeBytes(Data, offset, Size, inplace) && (inplace % 2 == 0)) {
            tensor.clamp_min_(min_val);
        }
        
        // Test out variant
        torch::Tensor out_tensor = torch::empty_like(tensor);
        torch::clamp_min_out(out_tensor, tensor, min_val);
        
        // Access results to ensure computation
        if (result.numel() > 0) {
            result.sum();
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}