#include<iostream>
#include<immintrin.h>
using namespace std;


int main(){
    __m256 r0, r1, r2;
    //__attribute__((aligned(256))) float a[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    //__attribute__((aligned(256))) float b[8] = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    //__attribute__((aligned(256))) float p[8]; 
    float a[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float b[8] = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    //float p[8]; 
    float *p = new float[8]; 
    //cout<<a[0]; 
    r0 = _mm256_loadu_ps(a);
    r1 = _mm256_loadu_ps(b);
    r2 = _mm256_add_ps(r0, r1);
    //cout<<r2[0];
    _mm256_storeu_ps(p, r2);
    for(int i = 0; i < 8; i++){
        cout<<p[i]<<" ";


    }
    
    return 1;
}
