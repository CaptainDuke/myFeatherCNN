//Tencent is pleased to support the open source community by making FeatherCNN available.

//Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.

//Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
//in compliance with the License. You may obtain a copy of the License at
//
//https://opensource.org/licenses/BSD-3-Clause
//
//Unless required by applicable law or agreed to in writing, software distributed
//under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//CONDITIONS OF ANY KIND, either express or implied. See the License for the
//specific language governing permissions and limitations under the License.

#include <iostream>
#include <booster/depthwise.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <immintrin.h>
//#include <arm_neon.h>

#ifdef __APPLE__
#else
#include <omp.h>
#endif
using namespace std;



template <bool fuseBias, bool fuseRelu>
void dwConv3s1(float *_output, const float *_input, int input_channels, int inw, int inh, float *_kernel, int group, int nThreads, float *bias_arr){
    std::cout<<"called successfully!"<<endl;

    int stridew = 1, strideh = 1;
    int kw = 3, kh = 3;
    int inch = input_channels;
    int outch = input_channels;
    int outw = (inw - kw)/stridew + 1;
    int outh = (inh - kh)/strideh + 1;
    const float* bias = bias_arr;
    const float* kernel = _kernel;
    //const float* input = _input;

    #pragma omp parallel for num_threads(opt.num_threads)
    for(int g = 0; g < inch; g++){



        float *outptr = _output + g*(outh*outw);
        float *outptr2 = outptr + outw;

        const float* img0 = _input + g*(inh*inw);
        const float* kernel0 = kernel + g*(kw*kh);

        float bias0;
        if(fuseBias)
            bias0 = bias[g];
        else
            bias0 = 0;

        const float* r0 = img0;
        const float* r1 = img0 + inw;
        const float* r2 = img0 + inw*2;
        const float* r3 = img0 + inw*3;

        const float* k0 = kernel0;
        const float* k1 = kernel0 + 3;
        const float* k2 = kernel0 + 6;
        
        int i = 0;

        for(; i + 1 < outh; i+=2){

            int remain = outw;

            for(; remain > 0; remain--){
                float sum = bias0;
                sum += r0[0] * k0[0];
                sum += r0[1] * k0[1];
                sum += r0[2] * k0[2];
                sum += r1[0] * k1[0];
                sum += r1[1] * k1[1];
                sum += r1[2] * k1[2];
                sum += r2[0] * k2[0];
                sum += r2[1] * k2[1];
                sum += r2[2] * k2[2];

                float sum2 = bias0;
                sum2 += r1[0] * k1[0];
                sum2 += r1[1] * k1[1];
                sum2 += r1[2] * k1[2];
                sum2 += r2[0] * k2[0];
                sum2 += r2[1] * k2[1];
                sum2 += r2[2] * k2[2];
                sum2 += r3[0] * k2[0];
                sum2 += r3[1] * k2[1];
                sum2 += r3[2] * k2[2];

                *outptr = sum;
                *outptr2 = sum2;

                r0++;
                r1++;
                r2++;
                r3++;
                outptr++;
                outptr2++;
            }

            r0 += 2 + inw;
            r1 += 2 + inw;
            r2 += 2 + inw;
            r3 += 2 + inw;

            outptr += outw;
            outptr2 += outw;
        }

        for(; i < outh; i++){
            int remain = outw;

            for(; remain > 0; remain--){
                float sum = bias0;
                sum += r0[0] * k0[0];
                sum += r0[1] * k0[1];
                sum += r0[2] * k0[2];
                sum += r1[0] * k1[0];
                sum += r1[1] * k1[1];
                sum += r1[2] * k1[2];
                sum += r2[0] * k2[0];
                sum += r2[1] * k2[1];
                sum += r2[2] * k2[2];
            
                *outptr = sum;

                r0++;
                r1++;
                r2++;
                outptr++;
            }
            
            r0 += 2;
            r1 += 2;
            r2 += 2;
        }


    }


}


template <bool fuseBias, bool fuseRelu>
void dwConv3s1Avx(float *_output, float *_input, int input_channels, int inw, int inh, float *_kernel, int group, int nThreads, float *bias_arr){
    int stridew = 1, strideh = 1;
    int kw = 3, kh = 3;
    int inch = input_channels;
    int outch = input_channels;
    int outw = (inw - kw)/stridew + 1;
    int outh = (inh - kh)/strideh + 1;
    float* bias = bias_arr;
    float* kernel = _kernel;

    __m256 vZero = _mm256_setzero_ps();

    __m256 vBias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for(int g = 0; g < inch; g++){
        float *outg = _output + g*(outh*outw);
        float* ing = _input + g*(inh*inw);
        float* kp = kernel + g*(kw*kh);


        if(fuseBias)
            vBias = _mm256_broadcast_ss(bias + g);
        __m256 k0 = _mm256_broadcast_ss(kp);
        // std::cout<<"k0:  "<<k0[1]<<" "<<k0[2];
        __m256 k1 = _mm256_broadcast_ss(kp + 1);
        __m256 k2 = _mm256_broadcast_ss(kp + 2);
        __m256 k3 = _mm256_broadcast_ss(kp + 3);
        __m256 k4 = _mm256_broadcast_ss(kp + 4);
        __m256 k5 = _mm256_broadcast_ss(kp + 5);
        __m256 k6 = _mm256_broadcast_ss(kp + 6);
        __m256 k7 = _mm256_broadcast_ss(kp + 7);
        __m256 k8 = _mm256_broadcast_ss(kp + 8);

        __m256 sum1, sum2, sum3, sum4;
        //__m256 k0_7, k3_
        int i = 0;
        
        // 情况1， 8个一起算
        for(; i + 1 < outh; i += 2){
            int nout = outw >> 3;
            int remain = outw & 7;
            float *r0 = ing + inw * i;
            float *r1 = ing + inw * (i + 1);
            float *r2 = ing + inw * (i + 2);
            float *r3 = ing + inw * (i + 3);

            float *og = outg + outw * i;
            float *og3 = og + outw;

            for(; nout > 0; nout--)
            {
                __m256 r00 = _mm256_loadu_ps(r0);
                __m256 r01 = _mm256_loadu_ps(r0 + 1);
                __m256 r02 = _mm256_loadu_ps(r0 + 2);

                __m256 r10 = _mm256_loadu_ps(r1);
                __m256 r11 = _mm256_loadu_ps(r1 + 1);
                __m256 r12 = _mm256_loadu_ps(r1 + 2);

                __m256 r20 = _mm256_loadu_ps(r2);
                __m256 r21 = _mm256_loadu_ps(r2 + 1);
                __m256 r22 = _mm256_loadu_ps(r2 + 2);

                __m256 r30 = _mm256_loadu_ps(r3);
                __m256 r31 = _mm256_loadu_ps(r3 + 1);
                __m256 r32 = _mm256_loadu_ps(r3 + 2);    

                sum1 = _mm256_mul_ps(r00, k0);
                sum1 = _mm256_fmadd_ps(r01, k1, sum1);
                sum1 = _mm256_fmadd_ps(r02, k2, sum1);
                sum1 = _mm256_fmadd_ps(r10, k3, sum1);
                sum1 = _mm256_fmadd_ps(r11, k4, sum1);
                sum1 = _mm256_fmadd_ps(r12, k5, sum1);
                sum1 = _mm256_fmadd_ps(r20, k6, sum1);
                sum1 = _mm256_fmadd_ps(r21, k7, sum1);
                sum1 = _mm256_fmadd_ps(r22, k8, sum1);

                sum2 = _mm256_mul_ps(r10, k0);
                sum2 = _mm256_fmadd_ps(r11, k1, sum2);
                sum2 = _mm256_fmadd_ps(r12, k2, sum2);
                sum2 = _mm256_fmadd_ps(r20, k3, sum2);
                sum2 = _mm256_fmadd_ps(r21, k4, sum2);
                sum2 = _mm256_fmadd_ps(r22, k5, sum2);
                sum2 = _mm256_fmadd_ps(r30, k6, sum2);
                sum2 = _mm256_fmadd_ps(r31, k7, sum2);
                sum2 = _mm256_fmadd_ps(r32, k8, sum2);   

                if(fuseBias){
                    sum1 = _mm256_add_ps(sum1, vBias);
                    sum2 = _mm256_add_ps(sum2, vBias);
                }
                if(fuseRelu)
                {
                    sum1 = _mm256_max_ps(sum1, vZero);
                    sum2 = _mm256_max_ps(sum2, vZero);
                }
                _mm256_storeu_ps(og, sum1);
                _mm256_storeu_ps(og3, sum2);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                r3 += 8;
                og += 8;
                og3 += 8;
            }

            //compute 2*4 in case of remain >=4, eg: 4,5,6,7
            for(; remain -3 > 0; remain -= 4)
            {
                __m256 r00 = _mm256_loadu_ps(r0);
                // __m256 r01 = _mm256_loadu2_m128(r0 + 1, r1 + 1);
                // __m256 r02 = _mm256_loadu2_m128(r0 + 2, r1 + 2); 

                // __m256 r30 = _mm256_loadu2_m128(r2, r3);
                // __m256 r31 = _mm256_loadu2_m128(r2 + 1, r3 + 1);
                // __m256 r32 = _mm256_loadu2_m128(r2 + 2, r3 + 2);     

                // sum1 = _mm256_mul_ps(r00, k0);
                // sum1 = _mm_fmadd_ps(r01, k1, sum1);
                // sum1 = _mm_fmadd_ps(r02, k1, sum1);
                // sum1 = _mm_fmadd_ps(r30, k1, sum1);
                // sum1 = _mm_fmadd_ps(r31, k1, sum1);
                // sum1 = _mm_fmadd_ps(r32, k1, sum1);




                /* 
                __m128 r00 = _mm_loadu_ps(r0);
                __m128 r00n = _mm_loadu_ps(r0 + 4);
                __m128 r01 = _mm_loadu_ps(r0 + 1);
                __m128 r02 = _mm_loadu_ps(r0 + 2);

                __m128 r10 = _mm_loadu_ps(r1);
                __m128 r10n = _mm_loadu_ps(r1 + 4);
                __m128 r11 = _mm_loadu_ps(r1 + 1);
                __m128 r12 = _mm_loadu_ps(r1 + 2);

                __m128 r20 = _mm_loadu_ps(r2);
                __m128 r20n = _mm_loadu_ps(r2 + 4);
                __m128 r21 = _mm_loadu_ps(r2 + 1);
                __m128 r22 = _mm_loadu_ps(r2 + 2);

                __m128 r30 = _mm_loadu_ps(r3);
                __m128 r30n = _mm_loadu_ps(r3 + 4);
                __m128 r31 = _mm_loadu_ps(r3 + 1);
                __m128 r32 = _mm_loadu_ps(r3 + 2);
 
                sum1 = _mm_mul_ps(r00, k0);
                sum1 = _mm_fmadd_ps(r01, k1, sum1);
                sum1 = _mm_fmadd_ps(r02, k2, sum1);
                sum1 = _mm_fmadd_ps(r10, k3, sum1);
                sum1 = _mm_fmadd_ps(r11, k4, sum1);
                sum1 = _mm_fmadd_ps(r12, k5, sum1);
                sum1 = _mm_fmadd_ps(r20, k6, sum1);
                sum1 = _mm_fmadd_ps(r21, k7, sum1);
                sum1 = _mm_fmadd_ps(r22, k8, sum1);

                sum2 = _mm_mul_ps(r10, k0);
                sum2 = _mm_fmadd_ps(r11, k1, sum2);
                sum2 = _mm_fmadd_ps(r12, k2, sum2);
                sum2 = _mm_fmadd_ps(r20, k3, sum2);
                sum2 = _mm_fmadd_ps(r21, k4, sum2);
                sum2 = _mm_fmadd_ps(r22, k5, sum2);
                sum2 = _mm_fmadd_ps(r30, k6, sum2);
                sum2 = _mm_fmadd_ps(r31, k7, sum2);
                sum2 = _mm_fmadd_ps(r32, k8, sum2);

                if(fuseBias){
                    sum1 = 
                }         
                */

            }



        }
        
  
        
    }


}




template <bool fuseBias, bool fuseRelu>
void globalDwConv(float *output, const float *input, int input_channels, int inw, int inh, float *kernel, int group, int nThreads, float *bias_arr)
{
    assert(group > 0 || input_channels % group == 0);
    int step = inw * inh;
    int block = input_channels / group;
    int groupKernelSize = inw * inh * group;

    for (int i = 0; i < input_channels; i++)
    {
        int k = i / group, u = i % group;
        output[i] = 0;
        for (int j = 0; j < step; j++)
        {
            output[i] += input[i * step + j] * kernel[k * groupKernelSize + u * step + j];
        }
        if (fuseBias)
        {
            output[i] += bias_arr[i];
        }
        if (fuseRelu)
        {
            output[i] = (output[i] > 0.f) ? output[i] : 0.f;
        }
    }

}

template <bool fuseBias, bool fuseRelu>
void dwConv_template(float *output, float *input, int input_channels, int inw, int inh, int stridew, int strideh, float *kernel, int kw, int kh, int group, int nThreads, float *bias_arr)
{

    std::cout<<"first called yep"<<endl;
    if ((kw == inw) && (kh == inh))
    {
        globalDwConv<fuseBias, fuseRelu>(output, input, input_channels, inw, inh, kernel, group, nThreads, bias_arr);
    }
    else if (kw == 3 && kh == 3 && stridew == 1 && strideh == 1){
        std::cout<<"second called:"<<endl;
        //dwConv3s1<fuseBias, fuseRelu>(output, input, input_channels, inw, inh, kernel, group, nThreads, bias_arr);
        dwConv3s1Avx<fuseBias, fuseRelu>(output, input, input_channels, inw, inh, kernel, group, nThreads, bias_arr);
    }

    else
    {
        int outw = (inw - kw) / stridew + 1; //for strided case in odd dimensions, should take the floor value as output dim.
        int outh = (inh - kh) / strideh + 1;

// #pragma omp parallel for num_threads(nThreads) schedule(static)
        //printf("dw param %d kernel %d %d stride %d %d input %d %d %d output %d %d\n", group, kh, kw, strideh, stridew, input_channels, inh, inw, outh, outw);
        for (int g = 0; g < group; ++g)
        {
            float *kp = kernel + kw * kh * g;
            float *outg = output + g * outw * outh;
            float *ing = input + g * inw * inh;
            for (int i = 0; i < outh; ++i)
            {
                for (int j = 0; j < outw; ++j)
                {
                    float *inp = ing + inw * (i * stridew) + (j * strideh);
                    float convSum = 0.f;
                    for (int m = 0; m < kh; m++)
                    {
                        for (int n = 0; n < kw; n++)
                        {
                            convSum += inp[m * inw + n] * kp[m * kw + n];
                        }
                    }
                    if (fuseBias)
                    {
                        convSum += bias_arr[g];
                    }
                    if (fuseRelu)
                    {
                        convSum = (convSum > 0.f) ? convSum : 0.f;
                    }
                    outg[j] = convSum;
                }
                outg += outw;
            }
        }
    }
}

template void dwConv_template<false, false>(float *, float *, int, int, int, int, int, float *, int, int, int, int, float *);
template void dwConv_template<false,  true>(float *, float *, int, int, int, int, int, float *, int, int, int, int, float *);
template void dwConv_template<true,  false>(float *, float *, int, int, int, int, int, float *, int, int, int, int, float *);
template void dwConv_template<true,   true>(float *, float *, int, int, int, int, int, float *, int, int, int, int, float *);



int main(){
    std::cout<<"test on PC windows"<<endl;
    //return 1;
    
    int inw = 10, inh = 10;

    int input_channels = 3;
    int kw = 3, kh = 3;
    int stridew = 1, strideh = 1;
    int group = 1, nThreads = 1;

    int outw = (inw - kw) / stridew + 1;
    int outh = (inh - kh) / strideh + 1;

    float output[input_channels * outw * outh], bias_arr[input_channels];
    float input[input_channels * inw * inh], kernel[input_channels * kw * kh];

    //std::cout<<sizeof(input)/sizeof(input[0]);
    for(int i = 0; i < sizeof(input)/sizeof(input[0]); i++){
        input[i] = i;
    }
    
    for(int i = 0; i < sizeof(kernel)/sizeof(kernel[0]); i++){
        kernel[i] = 1.0/9;
    }

    for(int i = 0; i < 3; i++){
        std::cout<<bias_arr[i]<<endl;
    }

    // for(int i = 0; i < input_channels; i++){
    //     for(int j = 0; j < inh; j++){
    //         for(int k = 0; k < inw; k++)
    //             std::cout<<input[i*(inh*inw) + j*(inw) + k]<<" ";
    //         std::cout<<endl;
    //     }
    //     std::cout<<endl<<endl;
    // }
    
    dwConv_template<false, false>(output, input, input_channels, inw, inh, stridew, strideh, kernel, kw, kh, group, nThreads, bias_arr);
    

    for(int i = 0; i < input_channels; i++){
        for(int j = 0; j < outh; j++){
            for(int k = 0; k < outw; k++)
                std::cout<<output[i*(outh*outw) + j*(outw) + k]<<" ";
            std::cout<<endl;
        }
        std::cout<<endl<<endl;
    }




    return 1;
    
}

