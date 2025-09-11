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
        
        // Need at least a few bytes to create a tensor and extract a vital name
        if (Size < 4) {
            return 0;
        }
        
        // Create a tensor to use with set_vital
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract a vital name from the remaining data
        if (offset < Size) {
            // Use some bytes to determine a vital name
            uint8_t name_selector = Data[offset++];
            
            // Select from a few predefined vital names
            std::vector<std::string> vital_names = {
                "accuracy", "loss", "precision", "recall", "f1_score", 
                "training_throughput", "inference_latency", "memory_usage"
            };
            
            std::string vital_name = vital_names[name_selector % vital_names.size()];
            
            // Extract a value for the vital
            double vital_value = 0.0;
            if (offset + sizeof(double) <= Size) {
                std::memcpy(&vital_value, Data + offset, sizeof(double));
                offset += sizeof(double);
            } else if (offset < Size) {
                // If we don't have enough bytes for a double, use what we have
                vital_value = static_cast<double>(Data[offset++]);
            }
            
            // Call torch::profiler::set_vital with the extracted name and value
            torch::profiler::set_vital(vital_name, vital_value);
            
            // Try setting a vital with a tensor value
            torch::profiler::set_vital(vital_name + "_tensor", tensor);
            
            // Try setting a vital with a string value
            std::string str_value = "test_value";
            if (offset < Size) {
                // Use remaining bytes to modify the string
                size_t len = std::min(Size - offset, str_value.size());
                for (size_t i = 0; i < len; i++) {
                    str_value[i] = static_cast<char>(Data[offset + i]);
                }
                offset += len;
            }
            torch::profiler::set_vital(vital_name + "_string", str_value);
            
            // Try setting a vital with a boolean value
            bool bool_value = false;
            if (offset < Size) {
                bool_value = (Data[offset++] % 2 == 0);
            }
            torch::profiler::set_vital(vital_name + "_bool", bool_value);
            
            // Try setting a vital with an integer value
            int64_t int_value = 0;
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&int_value, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
            } else if (offset < Size) {
                int_value = static_cast<int64_t>(Data[offset++]);
            }
            torch::profiler::set_vital(vital_name + "_int", int_value);
            
            // Try setting a vital with empty name
            torch::profiler::set_vital("", vital_value);
            
            // Try setting a vital with very long name
            std::string long_name(100, 'a');
            if (offset < Size) {
                long_name = std::string(Data[offset++] % 1000, 'a');
            }
            torch::profiler::set_vital(long_name, vital_value);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
