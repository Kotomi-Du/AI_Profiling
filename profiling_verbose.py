
# Creator: Sabharwal, Manuj R
# Modifier: Yaru, Du

'''
This script needs a csv file with the details of all layers of a single
inference. The name of the csv file is put in filenam variable. The variable
oplist contains which operations to be tested and an index for each of them.
Based on which ISA is being used, the output file name should be changed in 
the outnampart variable

This script will generate a single csv for each operation to be tested on.
The csv contains details like the time of execution of each layer.

To run the script, use the command line and don't run directly from IDLE.
'''


import os
import argparse
from glob import glob
import re
import csv
import subprocess

def create_header_forcsv(output_file):
    #write the header rows for the output file
    header = ['DNNL','Run','Device','Operator','ISA',' inference type','datatype','','','primitive','mb','g','ic','oc','ih','oh','kh','sh','dh','ph','iw','ow','kw','sw','dw','pw','is_int8','MACs','actual_time_ms','onednn_time_ms','ov_dnn_efficiency','ideal_time_ms','dnn_ideal_efficiency']
    header_str = ""
    for h in header:
        header_str += str(h) + ","
    header_str += "\n"
    output_file.write(header_str)
    

def parse_datatype(logstring):
    srctype = logstring.split("src_")[1].split(":")[0]
    weitype = logstring.split("wei_")[1].split(":")[0]
    dsttype = logstring.split("dst_")[1].split(":")[0]
    return srctype+weitype+dsttype

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Currently only works for convolution layer. if you use openvino benchmark_app, please set iter=1')
    parser.add_argument('-i', '--filefolder', type=str, default='logs', help='folder for log files(.txt) [ONEDNN_VERBOSE=1]')
    parser.add_argument('-c', '--cores', type=float, default='512', help='Enter number of EUs (GPU) or cores (CPU) in the machine')
    parser.add_argument('-f', '--frequency', type=float, default='2.1', help='frequency of machine(GHZ)')
    parser.add_argument('-d', '--device', type=str, default='gpu', help='gpu or cpu')
    parser.add_argument('--isa', type=str, default='unknown', help='the ISA used for CPU, avx2, avx512')
    #parser.add_argument('--debug', help="Optional. Don't show output.", action='store_true')
    args = parser.parse_args()
    
    if not os.path.exists(args.filefolder):
        print("Path is not exists: {}".format(args.filefolder))
        exit()
        
    files = glob(args.filefolder + "/*txt")
    num_cores = args.cores
    freq = args.frequency
    device = args.device
    isa_used = args.isa
    
    #change based on platform --> AVX2 - 8 and 32 and AVX512 16 and 64
    if isa_used.lower() == "avx512":
        ideal_macspersec_fp32 = 16*num_cores*freq*1000000000
        ideal_macspersec_int8 = 64*num_cores*freq*1000000000
    elif isa_used.lower() == "avx2":
        ideal_macspersec_fp32 = 8*num_cores*freq*1000000000
        ideal_macspersec_int8 = 32*num_cores*freq*1000000000
        
    
    ideal_macspersec_fp32_gpu = 16*num_cores*freq*1000000000
    ideal_macspersec_fp16_gpu = 128*num_cores*freq*1000000000
    ideal_macspersec_int8_gpu = 256*num_cores*freq*1000000000
    
    for input_file in files:

        print("Reading file %s" %input_file)
        output_file_name = input_file.split('.csv')[0] + '_processed.csv'
        print("--Writing file %s" %output_file_name)

        input_file = open(input_file, mode='r')
        output_file = open(output_file_name, mode='w')
        
        create_header_forcsv(output_file)
        

        for line in input_file:
            columns = line.split('\n')[0].split('\r')[0].split(',')
            # changing from dnnl_verbose to exec for the check since first 3 rows are info garbage
            if not 'onednn_verbose' in columns[0] and not 'dnnl_verbose' in columns[0]:
                continue
            if 'info' in columns[1]:
                continue
            
            regex1 = re.compile(r'mb(?P<mb>[0-9]+)*_ic(?P<ic>[0-9]+)*oc(?P<oc>[0-9]+)*_ih(?P<ih>[0-9]+)*oh(?P<oh>[0-9]+)*kh(?P<kh>[0-9]+)*sh(?P<sh>[0-9]+)*dh(?P<dh>[0-9]+)*ph(?P<ph>[0-9]+)*_iw(?P<iw>[0-9]+)*ow(?P<ow>[0-9]+)*kw(?P<kw>[0-9]+)*sw(?P<sw>[0-9]+)*dw(?P<dw>[0-9]+)*pw(?P<pw>[0-9]+)*')
            regex2 = re.compile(r'mb(?P<mb>[0-9]+)*_g(?P<g>[0-9]+)ic(?P<ic>[0-9]+)*oc(?P<oc>[0-9]+)*_ih(?P<ih>[0-9]+)*oh(?P<oh>[0-9]+)*kh(?P<kh>[0-9]+)*sh(?P<sh>[0-9]+)*dh(?P<dh>[0-9]+)*ph(?P<ph>[0-9]+)*_iw(?P<iw>[0-9]+)*ow(?P<ow>[0-9]+)*kw(?P<kw>[0-9]+)*sw(?P<sw>[0-9]+)*dw(?P<dw>[0-9]+)*pw(?P<pw>[0-9]+)*')
            regex3 = re.compile(r'g(?P<g>[0-9]+)mb(?P<mb>[0-9]+)*_ic(?P<ic>[0-9]+)*oc(?P<oc>[0-9]+)*_ih(?P<ih>[0-9]+)*oh(?P<oh>[0-9]+)*kh(?P<kh>[0-9]+)*sh(?P<sh>[0-9]+)*dh(?P<dh>[0-9]+)*ph(?P<ph>[0-9]+)*_iw(?P<iw>[0-9]+)*ow(?P<ow>[0-9]+)*kw(?P<kw>[0-9]+)*sw(?P<sw>[0-9]+)*dw(?P<dw>[0-9]+)*pw(?P<pw>[0-9]+)*')
            final_columns = columns
            #print(final_columns)
            
            # process only convolutions
            if columns[3] == 'convolution':
                #extract
                ptrn = columns[9]       #kernel column 
                if 'cpu' in columns[2]:
                    is_int8 = 'int8' in columns[4]  #jit_int8:* is in the 5th column
                if 'gpu' in columns[2]:
                    is_int8 = '8' in parse_datatype(columns[6])   # datatype configuration e.g. src_s8::blocked:aBcd32b:f0 wei_s8:p:blocked:ABcd8b8a4b:f0 bia_undef::undef::f0 dst_u8::blocked:acdb:f0
                    
                match = regex1.match(ptrn)
                if match:
                    conv_params = match.groupdict()
                    conv_params['g'] = 1
                else:
                    match = regex2.match(ptrn)
                    if match:
                        conv_params = match.groupdict()
                    else: 
                        match = regex3.match(ptrn)          #terrible bandaid fix for short term testing
                        conv_params = match.groupdict()
                    
                actual_time_ms = float(columns[10])

                #ideal compute
                macs = int(conv_params['mb'])*int(conv_params['ic'])*int(conv_params['oc'])*int(conv_params['oh'])*int(conv_params['ow'])*int(conv_params['kh'])*int(conv_params['kw'])/int(conv_params['g'])
                ideal_time = 0

                if 'cpu' in columns[2]:
                    if is_int8:
                        ideal_time_ms = (macs/ideal_macspersec_int8)*1000
                    else:
                        ideal_time_ms = (macs/ideal_macspersec_fp32)*1000
                elif 'gpu' in columns[2]:
                    if is_int8:
                        ideal_time_ms = (macs/ideal_macspersec_int8_gpu)*1000
                    else:
                        idx = columns[6].find("wei")
                        if columns[6][idx+4:idx+7] =="f16":
                            ideal_time_ms = (macs/ideal_macspersec_fp16_gpu)*1000
                        else:
                            ideal_time_ms = (macs/ideal_macspersec_fp32_gpu)*1000
                #efficiency = ideal_time_ms/actual_time_ms*100
                
                # benchdnn compute 
                strname = "benchdnn.exe --engine={} --conv --mode=p --cfg={} --dir=FWD_I  {}".format(device, parse_datatype(columns[6]),ptrn)
                result = subprocess.run(strname, stdout=subprocess.PIPE)
                out = str(result.stdout)
                idx = out.find('avg(ms):')
                dnn_time_ms = float(out[idx+8:-5])
                ov_dnn_efficiency= dnn_time_ms/actual_time_ms*100
                dnn_ideal_efficiency = ideal_time_ms/dnn_time_ms*100
                #get isa
                isa = "unknown"
                if "avx512" in columns[4]:
                    isa = "AVX512"
                elif "avx2" in columns[4]:
                    isa = "AVX2"
                elif "sse" in columns[4]:
                    isa = "SSE"
                elif "uni" in columns[4]:
                    isa = "UNI"
                elif "gemm" in columns[4]:
                    isa = "GEMM"
                if final_columns[2] == "cpu":
                    final_columns[4] = isa

                #output
                final_columns.pop()
                final_columns.append(conv_params['mb'])
                final_columns.append(conv_params['g'])
                final_columns.append(conv_params['ic'])
                final_columns.append(conv_params['oc'])
                final_columns.append(conv_params['ih'])
                final_columns.append(conv_params['oh'])
                final_columns.append(conv_params['kh'])
                final_columns.append(conv_params['sh'])
                final_columns.append(conv_params['dh'])
                final_columns.append(conv_params['ph'])
                final_columns.append(conv_params['iw'])
                final_columns.append(conv_params['ow'])
                final_columns.append(conv_params['kw'])
                final_columns.append(conv_params['sw'])
                final_columns.append(conv_params['dw'])
                final_columns.append(conv_params['pw'])
                final_columns.append(int(is_int8))
                final_columns.append(macs)
                final_columns.append(actual_time_ms)
                final_columns.append(dnn_time_ms)
                final_columns.append(ov_dnn_efficiency)
                final_columns.append(ideal_time_ms)
                final_columns.append(dnn_ideal_efficiency)
                
            output_str = ''
            for x in final_columns:
                output_str += str(x) + ','
            output_str += '\n'
            output_file.write(output_str)
        input_file.close()
        output_file.close()

    print("Processing completed!")