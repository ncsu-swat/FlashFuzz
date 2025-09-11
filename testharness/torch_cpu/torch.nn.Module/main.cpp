#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <torch/torch.h>

class SimpleModule : public torch::nn::Module {
public:
    SimpleModule() {
        linear1 = register_module("linear1", torch::nn::Linear(10, 8));
        linear2 = register_module("linear2", torch::nn::Linear(8, 4));
        dropout = register_module("dropout", torch::nn::Dropout(0.5));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(linear1->forward(x));
        x = dropout->forward(x);
        x = linear2->forward(x);
        return torch::log_softmax(x, 1);
    }

    torch::nn::Linear linear1{nullptr}, linear2{nullptr};
    torch::nn::Dropout dropout{nullptr};
};

class ConvModule : public torch::nn::Module {
public:
    ConvModule() {
        conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).stride(1).padding(1)));
        bn = register_module("bn", torch::nn::BatchNorm2d(16));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(bn->forward(conv->forward(x)));
        return torch::max_pool2d(x, 2);
    }

    torch::nn::Conv2d conv{nullptr};
    torch::nn::BatchNorm2d bn{nullptr};
};

class RecurrentModule : public torch::nn::Module {
public:
    RecurrentModule() {
        lstm = register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(10, 20).num_layers(2).bidirectional(true)));
        linear = register_module("linear", torch::nn::Linear(40, 5));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto lstm_out = lstm->forward(x);
        auto output = std::get<0>(lstm_out);
        auto last_output = output.select(0, output.size(0) - 1);
        return linear->forward(last_output);
    }

    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear linear{nullptr};
};

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        if (Size < 2) return 0;
        
        uint8_t module_type = Data[offset++] % 3;
        uint8_t batch_size = (Data[offset++] % 10) + 1;
        
        torch::Tensor input;
        torch::Tensor output;
        
        switch (module_type) {
            case 0: {
                auto simple_module = std::make_shared<SimpleModule>();
                
                if (offset < Size) {
                    try {
                        input = fuzzer_utils::createTensor(Data, Size, offset);
                        
                        if (input.dim() == 0) {
                            input = input.reshape({1, 10});
                        } else if (input.dim() == 1) {
                            if (input.size(0) != 10) {
                                input = input.reshape({1, -1});
                                if (input.size(1) != 10) {
                                    input = torch::randn({batch_size, 10});
                                }
                            } else {
                                input = input.reshape({1, 10});
                            }
                        } else {
                            if (input.size(-1) != 10) {
                                input = torch::randn({batch_size, 10});
                            }
                        }
                    } catch (...) {
                        input = torch::randn({batch_size, 10});
                    }
                } else {
                    input = torch::randn({batch_size, 10});
                }
                
                {
                    torch::NoGradGuard no_grad;
                    output = simple_module->forward(input);
                    
                    if (offset < Size) {
                        bool train_mode = Data[offset++] % 2 == 1;
                        if (train_mode) {
                            simple_module->train();
                        } else {
                            simple_module->eval();
                        }
                        
                        auto output2 = simple_module->forward(input);
                        
                        if (train_mode) {
                            simple_module->dropout->eval();
                            auto output3 = simple_module->forward(input);
                        }
                    }
                }
                break;
            }
            case 1: {
                auto conv_module = std::make_shared<ConvModule>();
                
                if (offset < Size) {
                    try {
                        input = fuzzer_utils::createTensor(Data, Size, offset);
                        
                        if (input.dim() < 4) {
                            input = torch::randn({batch_size, 3, 32, 32});
                        } else if (input.size(1) != 3) {
                            input = input.reshape({-1, 3, input.size(2), input.size(3)});
                        }
                    } catch (...) {
                        input = torch::randn({batch_size, 3, 32, 32});
                    }
                } else {
                    input = torch::randn({batch_size, 3, 32, 32});
                }
                
                {
                    torch::NoGradGuard no_grad;
                    output = conv_module->forward(input);
                    
                    if (offset < Size) {
                        bool train_mode = Data[offset++] % 2 == 1;
                        if (train_mode) {
                            conv_module->train();
                        } else {
                            conv_module->eval();
                        }
                        
                        auto output2 = conv_module->forward(input);
                    }
                }
                break;
            }
            case 2: {
                auto rnn_module = std::make_shared<RecurrentModule>();
                
                if (offset < Size) {
                    try {
                        input = fuzzer_utils::createTensor(Data, Size, offset);
                        
                        if (input.dim() < 3) {
                            input = torch::randn({5, batch_size, 10});
                        } else if (input.size(2) != 10) {
                            input = input.reshape({input.size(0), input.size(1), 10});
                        }
                    } catch (...) {
                        input = torch::randn({5, batch_size, 10});
                    }
                } else {
                    input = torch::randn({5, batch_size, 10});
                }
                
                {
                    torch::NoGradGuard no_grad;
                    output = rnn_module->forward(input);
                    
                    if (offset < Size) {
                        bool train_mode = Data[offset++] % 2 == 1;
                        if (train_mode) {
                            rnn_module->train();
                        } else {
                            rnn_module->eval();
                        }
                        
                        auto output2 = rnn_module->forward(input);
                    }
                }
                break;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
