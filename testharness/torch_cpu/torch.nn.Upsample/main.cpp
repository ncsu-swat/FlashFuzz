#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Create input tensor
        if (offset >= Size) return 0;
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse mode from input data
        torch::nn::functional::InterpolateFuncOptions::mode_t mode = torch::kNearest;
        if (offset < Size) {
            uint8_t mode_selector = Data[offset++];
            if (mode_selector % 3 == 0) {
                mode = torch::kNearest;
            } else if (mode_selector % 3 == 1) {
                mode = torch::kLinear;
            } else {
                mode = torch::kBilinear;
            }
        }
        
        // Parse align_corners flag
        bool align_corners = false;
        if (offset < Size) {
            align_corners = Data[offset++] & 1;
        }
        
        // Parse scale factors or output size
        bool use_scale_factor = true;
        if (offset < Size) {
            use_scale_factor = Data[offset++] & 1;
        }
        
        // Get input dimensions
        auto input_dims = input.sizes();
        int64_t dim = input_dims.size();
        
        // Create interpolate options
        torch::nn::functional::InterpolateFuncOptions options;
        options.mode(mode);
        
        // Set align_corners if applicable (only for some modes)
        if (mode != torch::kNearest) {
            options.align_corners(align_corners);
        }
        
        // Parse scale factors or output size based on input dimensions
        if (use_scale_factor && dim > 0) {
            std::vector<double> scale_factors;
            for (int i = 0; i < dim - 2; i++) {
                double scale = 1.0;
                if (offset + sizeof(double) <= Size) {
                    memcpy(&scale, Data + offset, sizeof(double));
                    offset += sizeof(double);
                }
                scale_factors.push_back(scale);
            }
            
            if (!scale_factors.empty()) {
                options.scale_factor(scale_factors);
            }
        } else if (dim > 0) {
            std::vector<int64_t> output_size;
            for (int i = 0; i < dim - 2; i++) {
                int64_t size_val = 1;
                if (offset + sizeof(int64_t) <= Size) {
                    memcpy(&size_val, Data + offset, sizeof(int64_t));
                    offset += sizeof(int64_t);
                    // Ensure size is positive
                    size_val = std::abs(size_val) % 100 + 1;
                }
                output_size.push_back(size_val);
            }
            
            if (!output_size.empty()) {
                options.size(output_size);
            }
        }
        
        // Apply interpolate operation
        torch::Tensor output;
        try {
            output = torch::nn::functional::interpolate(input, options);
        } catch (const c10::Error& e) {
            // Catch PyTorch-specific errors
            return 0;
        }
        
        // Perform some operations on the output to ensure it's used
        if (output.defined()) {
            auto sum = output.sum();
            auto mean = output.mean();
            auto max_val = output.max();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
