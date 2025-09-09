#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>
#include <string>
#include <vector>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 1) return 0;
        
        // Extract fuzzing parameters
        uint8_t config_byte = consume_uint8_t(Data, Size, offset);
        
        // Test different parameter combinations based on config_byte
        switch (config_byte % 8) {
            case 0: {
                // Test precision parameter
                if (offset + sizeof(int32_t) <= Size) {
                    int32_t precision = consume_int32_t(Data, Size, offset);
                    // Clamp precision to reasonable range to avoid extreme values
                    precision = std::max(-10, std::min(precision, 50));
                    torch::set_printoptions(torch::TensorPrintOptions().precision(precision));
                    
                    // Create a test tensor to verify the setting works
                    auto tensor = torch::randn({2, 3});
                    std::ostringstream oss;
                    oss << tensor;
                }
                break;
            }
            case 1: {
                // Test threshold parameter
                if (offset + sizeof(int32_t) <= Size) {
                    int32_t threshold = consume_int32_t(Data, Size, offset);
                    // Clamp threshold to reasonable range
                    threshold = std::max(0, std::min(threshold, 100000));
                    torch::set_printoptions(torch::TensorPrintOptions().threshold(threshold));
                    
                    // Create a test tensor to verify the setting works
                    auto tensor = torch::arange(20);
                    std::ostringstream oss;
                    oss << tensor;
                }
                break;
            }
            case 2: {
                // Test edgeitems parameter
                if (offset + sizeof(int32_t) <= Size) {
                    int32_t edgeitems = consume_int32_t(Data, Size, offset);
                    // Clamp edgeitems to reasonable range
                    edgeitems = std::max(0, std::min(edgeitems, 100));
                    torch::set_printoptions(torch::TensorPrintOptions().edgeitems(edgeitems));
                    
                    // Create a test tensor to verify the setting works
                    auto tensor = torch::arange(50);
                    std::ostringstream oss;
                    oss << tensor;
                }
                break;
            }
            case 3: {
                // Test linewidth parameter
                if (offset + sizeof(int32_t) <= Size) {
                    int32_t linewidth = consume_int32_t(Data, Size, offset);
                    // Clamp linewidth to reasonable range
                    linewidth = std::max(1, std::min(linewidth, 1000));
                    torch::set_printoptions(torch::TensorPrintOptions().linewidth(linewidth));
                    
                    // Create a test tensor to verify the setting works
                    auto tensor = torch::randn({5, 10});
                    std::ostringstream oss;
                    oss << tensor;
                }
                break;
            }
            case 4: {
                // Test sci_mode parameter
                if (offset < Size) {
                    bool sci_mode = (Data[offset] % 2) == 0;
                    offset++;
                    torch::set_printoptions(torch::TensorPrintOptions().sci_mode(sci_mode));
                    
                    // Create a test tensor with scientific notation candidates
                    auto tensor = torch::tensor({1e-10, 1e10, 0.000001, 1000000.0});
                    std::ostringstream oss;
                    oss << tensor;
                }
                break;
            }
            case 5: {
                // Test multiple parameters together
                if (offset + 3 * sizeof(int32_t) + 1 <= Size) {
                    int32_t precision = consume_int32_t(Data, Size, offset);
                    int32_t threshold = consume_int32_t(Data, Size, offset);
                    int32_t linewidth = consume_int32_t(Data, Size, offset);
                    bool sci_mode = (consume_uint8_t(Data, Size, offset) % 2) == 0;
                    
                    // Clamp values
                    precision = std::max(0, std::min(precision, 20));
                    threshold = std::max(1, std::min(threshold, 10000));
                    linewidth = std::max(10, std::min(linewidth, 500));
                    
                    torch::set_printoptions(torch::TensorPrintOptions()
                        .precision(precision)
                        .threshold(threshold)
                        .linewidth(linewidth)
                        .sci_mode(sci_mode));
                    
                    // Create various test tensors
                    auto tensor1 = torch::randn({3, 4});
                    auto tensor2 = torch::arange(100);
                    std::ostringstream oss1, oss2;
                    oss1 << tensor1;
                    oss2 << tensor2;
                }
                break;
            }
            case 6: {
                // Test edge cases with extreme but valid values
                if (offset + sizeof(int32_t) <= Size) {
                    int32_t param_choice = consume_int32_t(Data, Size, offset) % 4;
                    
                    switch (param_choice) {
                        case 0:
                            // Very high precision
                            torch::set_printoptions(torch::TensorPrintOptions().precision(15));
                            break;
                        case 1:
                            // Very low threshold
                            torch::set_printoptions(torch::TensorPrintOptions().threshold(1));
                            break;
                        case 2:
                            // Very high linewidth
                            torch::set_printoptions(torch::TensorPrintOptions().linewidth(500));
                            break;
                        case 3:
                            // Zero edgeitems
                            torch::set_printoptions(torch::TensorPrintOptions().edgeitems(0));
                            break;
                    }
                    
                    // Test with various tensor types
                    auto float_tensor = torch::randn({5, 5});
                    auto int_tensor = torch::randint(0, 100, {5, 5});
                    auto large_tensor = torch::arange(1000);
                    
                    std::ostringstream oss1, oss2, oss3;
                    oss1 << float_tensor;
                    oss2 << int_tensor;
                    oss3 << large_tensor;
                }
                break;
            }
            case 7: {
                // Test rapid successive calls to set_printoptions
                for (int i = 0; i < 5 && offset < Size; i++) {
                    uint8_t param = consume_uint8_t(Data, Size, offset);
                    int precision = (param % 10) + 1;
                    torch::set_printoptions(torch::TensorPrintOptions().precision(precision));
                    
                    // Quick tensor creation and printing
                    auto tensor = torch::tensor({1.23456789});
                    std::ostringstream oss;
                    oss << tensor;
                }
                break;
            }
        }
        
        // Always reset to default at the end to avoid affecting other tests
        torch::set_printoptions(torch::TensorPrintOptions());
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}