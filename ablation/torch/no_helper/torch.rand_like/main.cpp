#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeBytes(const uint8_t* data, size_t size, size_t& offset, T& value) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&value, data + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) return 0;
    
    try {
        size_t offset = 0;
        
        // Consume parameters for tensor creation
        uint8_t ndim;
        if (!consumeBytes(data, size, offset, ndim)) return 0;
        ndim = (ndim % 5) + 1; // 1-5 dimensions
        
        // Build shape
        std::vector<int64_t> shape;
        for (int i = 0; i < ndim; i++) {
            uint8_t dim_size;
            if (!consumeBytes(data, size, offset, dim_size)) {
                shape.push_back(1);
            } else {
                // Allow 0-sized dimensions and various sizes
                shape.push_back(dim_size % 10); 
            }
        }
        
        // Select dtype for input tensor
        uint8_t dtype_selector;
        if (!consumeBytes(data, size, offset, dtype_selector)) dtype_selector = 0;
        
        torch::ScalarType input_dtype;
        switch (dtype_selector % 8) {
            case 0: input_dtype = torch::kFloat32; break;
            case 1: input_dtype = torch::kFloat64; break;
            case 2: input_dtype = torch::kInt32; break;
            case 3: input_dtype = torch::kInt64; break;
            case 4: input_dtype = torch::kInt8; break;
            case 5: input_dtype = torch::kUInt8; break;
            case 6: input_dtype = torch::kFloat16; break;
            case 7: input_dtype = torch::kBool; break;
            default: input_dtype = torch::kFloat32;
        }
        
        // Create input tensor
        auto options = torch::TensorOptions().dtype(input_dtype);
        
        // Device selection
        uint8_t device_selector;
        if (consumeBytes(data, size, offset, device_selector)) {
            if (device_selector % 2 == 0) {
                options = options.device(torch::kCPU);
            } else if (torch::cuda::is_available()) {
                options = options.device(torch::kCUDA);
            }
        }
        
        torch::Tensor input_tensor;
        try {
            input_tensor = torch::zeros(shape, options);
        } catch (...) {
            // If tensor creation fails, try with smaller shape
            shape = {2, 3};
            input_tensor = torch::zeros(shape, torch::kFloat32);
        }
        
        // Fill with some data based on fuzzer input
        if (offset < size && input_tensor.numel() > 0) {
            uint8_t fill_val;
            consumeBytes(data, size, offset, fill_val);
            input_tensor.fill_(static_cast<float>(fill_val) / 255.0f);
        }
        
        // Test rand_like with various parameter combinations
        uint8_t param_selector;
        if (!consumeBytes(data, size, offset, param_selector)) param_selector = 0;
        
        // Basic rand_like
        torch::Tensor result1 = torch::rand_like(input_tensor);
        
        // rand_like with different dtype
        if (param_selector & 0x01) {
            uint8_t output_dtype_sel;
            if (!consumeBytes(data, size, offset, output_dtype_sel)) output_dtype_sel = 0;
            
            torch::ScalarType output_dtype;
            switch (output_dtype_sel % 4) {
                case 0: output_dtype = torch::kFloat32; break;
                case 1: output_dtype = torch::kFloat64; break;
                case 2: output_dtype = torch::kFloat16; break;
                case 3: output_dtype = torch::kBFloat16; break;
                default: output_dtype = torch::kFloat32;
            }
            
            try {
                torch::Tensor result2 = torch::rand_like(input_tensor, output_dtype);
            } catch (...) {
                // Silently ignore dtype conversion errors
            }
        }
        
        // rand_like with requires_grad
        if (param_selector & 0x02) {
            try {
                auto result3 = torch::rand_like(input_tensor, torch::requires_grad(true));
            } catch (...) {
                // Ignore gradient-related errors
            }
        }
        
        // rand_like with memory format
        if (param_selector & 0x04) {
            uint8_t mem_format_sel;
            if (!consumeBytes(data, size, offset, mem_format_sel)) mem_format_sel = 0;
            
            torch::MemoryFormat mem_format;
            switch (mem_format_sel % 3) {
                case 0: mem_format = torch::MemoryFormat::Contiguous; break;
                case 1: mem_format = torch::MemoryFormat::ChannelsLast; break;
                case 2: mem_format = torch::MemoryFormat::Preserve; break;
                default: mem_format = torch::MemoryFormat::Preserve;
            }
            
            try {
                torch::Tensor result4 = torch::rand_like(input_tensor, torch::MemoryFormat(mem_format));
            } catch (...) {
                // Ignore memory format errors
            }
        }
        
        // Test with strided tensor
        if (param_selector & 0x08 && input_tensor.numel() > 1) {
            try {
                auto strided = input_tensor.as_strided({1}, {2});
                torch::Tensor result5 = torch::rand_like(strided);
            } catch (...) {
                // Ignore striding errors
            }
        }
        
        // Test with transposed tensor
        if (param_selector & 0x10 && input_tensor.dim() >= 2) {
            try {
                auto transposed = input_tensor.transpose(0, 1);
                torch::Tensor result6 = torch::rand_like(transposed);
            } catch (...) {
                // Ignore transpose errors
            }
        }
        
        // Test with view
        if (param_selector & 0x20 && input_tensor.numel() > 0) {
            try {
                auto viewed = input_tensor.view({-1});
                torch::Tensor result7 = torch::rand_like(viewed);
            } catch (...) {
                // Ignore view errors
            }
        }
        
        // Test edge case: empty tensor
        if (param_selector & 0x40) {
            try {
                torch::Tensor empty_tensor = torch::empty({0, 3}, input_tensor.options());
                torch::Tensor result8 = torch::rand_like(empty_tensor);
            } catch (...) {
                // Ignore empty tensor errors
            }
        }
        
        // Test with complex combinations
        if (offset < size - 2) {
            try {
                uint8_t combo;
                consumeBytes(data, size, offset, combo);
                
                auto opts = input_tensor.options();
                if (combo & 0x01) opts = opts.dtype(torch::kFloat64);
                if (combo & 0x02) opts = opts.requires_grad(true);
                
                torch::Tensor result9 = torch::rand_like(input_tensor, opts);
            } catch (...) {
                // Ignore combination errors
            }
        }
        
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        std::cout << "Exception caught: unknown exception" << std::endl;
        return -1;
    }
    
    return 0;
}