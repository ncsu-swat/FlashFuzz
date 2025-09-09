tensorflow::DataType parseDataType(uint8_t selector) {
    // You need to abandon some data types based on the document to avoid invalid inputs
  tensorflow::DataType dtype; 
  switch (selector % 23) {  
    case 0:
      dtype = DT_FLOAT;
      break;
    case 1:
      dtype = DT_DOUBLE;
      break;
    case 2:
      dtype = DT_INT32;
      break;
    case 3:
      dtype = DT_UINT8;
      break;
    case 4:
      dtype = DT_INT16;
      break;
    case 5:
      dtype = DT_INT8;
      break;
    case 6:
      dtype = DT_STRING;
      break;
    case 7:
      dtype = DT_COMPLEX64; // float_complex
      break;
    case 8:
      dtype = DT_INT64;
      break;
    case 9:
      dtype = DT_BOOL;
      break;
    case 10:
      dtype = DT_QINT8; // quantized_int8
      break;
    case 11:
      dtype = DT_QUINT8; // quantized_uint8
      break;
    case 12:
      dtype = DT_QINT32; // quantized_int32
      break;
    case 13:
      dtype = DT_BFLOAT16; // bfloat16
      break;
    case 14:
      dtype = DT_QINT16; // quantized_int16
      break;
    case 15:
      dtype = DT_QUINT16; // quantized_uint16
      break;
    case 16:
      dtype = DT_UINT16;
      break;
    case 17:
      dtype = DT_COMPLEX128; // double_complex
      break;
    case 18:
      dtype = DT_HALF; // float16
      break;
    case 19:
      dtype = DT_UINT32;
      break;
    case 20:
      dtype = DT_UINT64;
      break;
  }

  return dtype;
}