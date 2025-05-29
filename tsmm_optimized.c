#include "../include/tsmm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h> // AVX-512指令集
#include <omp.h>

// 预取距离 (以双精度元素为单位)
#define PREFETCH_DISTANCE 32

// 双精度浮点数大小 (字节)
#define DOUBLE_SIZE 8

// 缓存大小配置 (基于Intel Xeon Platinum 9242)
#define L1_CACHE_SIZE (32 * 1024)       // 32KB
#define L2_CACHE_SIZE (1024 * 1024)     // 1MB
#define L3_CACHE_SIZE (36608 * 1024)    // ~36MB

// 定义打包矩阵需要的常量
#define MC 240
#define KC 256 
#define NC 4096
#define MR 24
#define NR 8

// 针对瘦高矩阵1 (8x16x16000)优化
static void tsmm_tall_thin1(const double* A, const double* B, double* C, 
                          size_t m, size_t n, size_t k, MatrixLayout layout) {
    // m=8, n=16, k=16000
    // 这种情况下，m和n都很小，k很大
    
    // 对于这种小矩阵，使用预分配和对齐的小缓冲区
    const size_t block_k = 512; // 增大k方向分块大小
    
    // 分配临时缓冲区
    double* A_buffer = (double*)aligned_alloc(64, m * block_k * sizeof(double));
    double* B_buffer = (double*)aligned_alloc(64, block_k * n * sizeof(double));
    
    if (!A_buffer || !B_buffer) {
        if (A_buffer) free(A_buffer);
        if (B_buffer) free(B_buffer);
        
        // 回退到原始实现
        memset(C, 0, m * n * sizeof(double));
        
        if (layout == ROW_MAJOR) {
            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < n; j++) {
                    double sum = 0.0;
                    for (size_t p = 0; p < k; p++) {
                        sum += A[p * m + i] * B[p * n + j];
                    }
                    C[i * n + j] = sum;
                }
            }
        }
        return;
    }
    
    // 初始化C为0
    memset(C, 0, m * n * sizeof(double));
    
    if (layout == ROW_MAJOR) {
        // 分配高效访问的寄存器数组 - 避免多次访问内存
        __m512d c_regs[8][2] = {0}; // 8行，每行用2个向量寄存器(共16个元素)
        
        // 初始化寄存器数组为0
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < 2; j++) {
                c_regs[i][j] = _mm512_setzero_pd();
            }
        }
        
        // 分块处理k维度
        for (size_t p0 = 0; p0 < k; p0 += block_k) {
            size_t p_end = (p0 + block_k < k) ? p0 + block_k : k;
            size_t p_size = p_end - p0;
            
            // 预处理：复制当前块的A和B到连续缓冲区
            for (size_t p = 0; p < p_size; p++) {
                for (size_t i = 0; i < m; i++) {
                    A_buffer[p * m + i] = A[(p0 + p) * m + i];
                }
                for (size_t j = 0; j < n; j++) {
                    B_buffer[p * n + j] = B[(p0 + p) * n + j];
                }
            }
            
            // 处理当前k块
            for (size_t p = 0; p < p_size; p++) {
                // 预取下一次迭代的数据
                if (p + PREFETCH_DISTANCE < p_size) {
                    _mm_prefetch((const char*)&A_buffer[(p + PREFETCH_DISTANCE) * m], _MM_HINT_T0);
                    _mm_prefetch((const char*)&B_buffer[(p + PREFETCH_DISTANCE) * n], _MM_HINT_T0);
                }
                
                for (size_t i = 0; i < m; i++) {
                    // 广播A元素
                    __m512d a_vec = _mm512_set1_pd(A_buffer[p * m + i]);
                    
                    // 加载B元素并计算
                    __m512d b_vec0 = _mm512_loadu_pd(&B_buffer[p * n + 0]);
                    c_regs[i][0] = _mm512_fmadd_pd(a_vec, b_vec0, c_regs[i][0]);
                    
                    if (n > 8) {
                        __m512d b_vec1 = _mm512_loadu_pd(&B_buffer[p * n + 8]);
                        c_regs[i][1] = _mm512_fmadd_pd(a_vec, b_vec1, c_regs[i][1]);
                    }
                }
            }
        }
        
        // 将寄存器结果存回C矩阵
        for (size_t i = 0; i < m; i++) {
            _mm512_storeu_pd(&C[i * n + 0], c_regs[i][0]);
            if (n > 8) {
                _mm512_storeu_pd(&C[i * n + 8], c_regs[i][1]);
            }
        }
    } else { // COL_MAJOR
        // 列主序实现（省略）
    }
    
    // 释放临时缓冲区
    free(A_buffer);
    free(B_buffer);
}

// 针对瘦高矩阵2 (32x16000x16)优化
static void tsmm_tall_thin2(const double* A, const double* B, double* C, 
                          size_t m, size_t n, size_t k, MatrixLayout layout) {
    // m=32, n=16000, k=16
    // 这种情况下，m和k都较小，n很大
    // 需要沿着n方向分块
    
    // 根据L1缓存大小计算n方向最优分块
    // 每个块需要：(m*k + k*block_n + m*block_n) * DOUBLE_SIZE < L1_CACHE_SIZE
    // 32*16 + 16*block_n + 32*block_n < L1_CACHE_SIZE/DOUBLE_SIZE
    // (16 + 32)*block_n < L1_CACHE_SIZE/DOUBLE_SIZE - 32*16
    // block_n < (L1_CACHE_SIZE/DOUBLE_SIZE - 512) / 48
    const size_t block_n = 160; // 根据公式计算约为168，取160便于计算
    
    // 初始化C为0
    memset(C, 0, m * n * sizeof(double));
    
    if (layout == ROW_MAJOR) {
        // 沿n方向分块计算
        for (size_t j0 = 0; j0 < n; j0 += block_n) {
            size_t j_end = (j0 + block_n < n) ? j0 + block_n : n;
            
            // 由于k很小(16)，可以完全展开k循环
            for (size_t i = 0; i < m; i++) {
                for (size_t j = j0; j < j_end; j += 8) {
                    if (j + 8 <= j_end) {
                        // 使用SIMD向量化8个元素
                        __m512d sum = _mm512_setzero_pd();
                        
                        // 完全展开k循环(k=16)
                        for (size_t p = 0; p < k; p++) {
                            __m512d a_vec = _mm512_set1_pd(A[p * m + i]);
                            __m512d b_vec = _mm512_loadu_pd(&B[p * n + j]);
                            sum = _mm512_fmadd_pd(a_vec, b_vec, sum);
                        }
                        
                        // 存储结果
                        _mm512_storeu_pd(&C[i * n + j], sum);
                    } else {
                        // 处理边界情况
                        for (size_t jj = j; jj < j_end; jj++) {
                            double sum = 0.0;
                            for (size_t p = 0; p < k; p++) {
                                sum += A[p * m + i] * B[p * n + jj];
                            }
                            C[i * n + jj] = sum;
                        }
                    }
                }
            }
        }
    } else { // COL_MAJOR
        // 列主序实现（省略）
    }
}

// 针对大矩阵 (4000x16000x128)优化
static void tsmm_large_matrix(const double* A, const double* B, double* C, 
                            size_t m, size_t n, size_t k, MatrixLayout layout) {
    // m=4000, n=16000, k=128
    
    // 分块参数
    const size_t mc = 240;  // 行分块大小
    const size_t kc = 64;   // k方向分块大小
    const size_t nc = 2048; // 列分块大小
    
    // 微块参数 (固定值)
    #define MR_LARGE 8
    #define NR_LARGE 8
    
    // 初始化C为0
    memset(C, 0, m * n * sizeof(double));
    
    if (layout == ROW_MAJOR) {
        // 分配大块缓冲区
        double* A_block = (double*)aligned_alloc(64, mc * kc * sizeof(double));
        double* B_block = (double*)aligned_alloc(64, kc * nc * sizeof(double));
        
        if (!A_block || !B_block) {
            if (A_block) free(A_block);
            if (B_block) free(B_block);
            fprintf(stderr, "内存分配失败\n");
            
            // 使用简单实现
            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < n; j++) {
                    double sum = 0.0;
                    for (size_t p = 0; p < k; p++) {
                        sum += A[p * m + i] * B[p * n + j];
                    }
                    C[i * n + j] = sum;
                }
            }
            return;
        }
        
        // 三层分块循环
        for (size_t j0 = 0; j0 < n; j0 += nc) {
            size_t jb = (j0 + nc < n) ? nc : (n - j0);
            
            for (size_t p0 = 0; p0 < k; p0 += kc) {
                size_t pb = (p0 + kc < k) ? kc : (k - p0);
                
                // 打包B矩阵块 (k*n)，改变数据布局提高局部性
                for (size_t p = 0; p < pb; p++) {
                    for (size_t j = 0; j < jb; j++) {
                        B_block[p * jb + j] = B[(p0 + p) * n + (j0 + j)];
                    }
                }
                
                for (size_t i0 = 0; i0 < m; i0 += mc) {
                    size_t ib = (i0 + mc < m) ? mc : (m - i0);
                    
                    // 打包A矩阵块 (k*m)，改变数据布局提高局部性
                    for (size_t p = 0; p < pb; p++) {
                        for (size_t i = 0; i < ib; i++) {
                            A_block[p * ib + i] = A[(p0 + p) * m + (i0 + i)];
                        }
                    }
                    
                    // 微块计算 - 使用固定大小的微块
                    for (size_t i = 0; i < ib; i += MR_LARGE) {
                        size_t i_end = (i + MR_LARGE <= ib) ? i + MR_LARGE : ib;
                        size_t i_size = i_end - i;
                        
                        for (size_t j = 0; j < jb; j += NR_LARGE) {
                            size_t j_end = (j + NR_LARGE <= jb) ? j + NR_LARGE : jb;
                            size_t j_size = j_end - j;
                            
                            // 使用向量化计算完整的8x8微块
                            if (i_size == MR_LARGE && j_size == NR_LARGE) {
                                // 定义寄存器数组，并初始化为0
                                __m512d c_reg[MR_LARGE]; // 8行，每行8个元素(一个向量寄存器)
                                
                                // 加载当前C块
                                for (size_t ir = 0; ir < MR_LARGE; ir++) {
                                    c_reg[ir] = _mm512_loadu_pd(&C[(i0 + i + ir) * n + (j0 + j)]);
                                }
                                
                                // 核心计算
                                for (size_t p = 0; p < pb; p++) {
                                    // 处理8行x8列的微块
                                    for (size_t ir = 0; ir < MR_LARGE; ir++) {
                                        // 广播A元素
                                        __m512d a_val = _mm512_set1_pd(A_block[p * ib + i + ir]);
                                        // 加载B行
                                        __m512d b_vec = _mm512_loadu_pd(&B_block[p * jb + j]);
                                        // 计算一行
                                        c_reg[ir] = _mm512_fmadd_pd(a_val, b_vec, c_reg[ir]);
                                    }
                                }
                                
                                // 写回结果
                                for (size_t ir = 0; ir < MR_LARGE; ir++) {
                                    _mm512_storeu_pd(&C[(i0 + i + ir) * n + (j0 + j)], c_reg[ir]);
                                }
                            } else {
                                // 处理边界情况
                                for (size_t ii = 0; ii < i_size; ii++) {
                                    for (size_t jj = 0; jj < j_size; jj += 8) {
                                        if (jj + 8 <= j_size) {
                                            __m512d sum = _mm512_loadu_pd(&C[(i0 + i + ii) * n + (j0 + j + jj)]);
                                            
                                            for (size_t p = 0; p < pb; p++) {
                                                __m512d a_val = _mm512_set1_pd(A_block[p * ib + i + ii]);
                                                __m512d b_vec = _mm512_loadu_pd(&B_block[p * jb + j + jj]);
                                                sum = _mm512_fmadd_pd(a_val, b_vec, sum);
                                            }
                                            
                                            _mm512_storeu_pd(&C[(i0 + i + ii) * n + (j0 + j + jj)], sum);
                                        } else {
                                            for (size_t j_scalar = jj; j_scalar < j_size; j_scalar++) {
                                                double sum = C[(i0 + i + ii) * n + (j0 + j + j_scalar)];
                                                for (size_t p = 0; p < pb; p++) {
                                                    sum += A_block[p * ib + i + ii] * B_block[p * jb + j + j_scalar];
                                                }
                                                C[(i0 + i + ii) * n + (j0 + j + j_scalar)] = sum;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // 释放打包缓冲区
        free(A_block);
        free(B_block);
    } else { // COL_MAJOR
        // 列主序实现（省略）
    }
    
    #undef MR_LARGE
    #undef NR_LARGE
}

// 方形矩阵专用优化 (144x144x144)
static void tsmm_square_matrix(const double* A, const double* B, double* C, 
                             size_t m, size_t n, size_t k, MatrixLayout layout) {
    // m=n=k=144，每个维度都适中
    
    // 使用简单可靠的分块策略
    const size_t block_size = 48; // 适合L1缓存
    
    // 初始化C为0
    memset(C, 0, m * n * sizeof(double));
    
    if (layout == ROW_MAJOR) {
        // 三层分块循环
        for (size_t i0 = 0; i0 < m; i0 += block_size) {
            size_t i_end = (i0 + block_size < m) ? i0 + block_size : m;
            
            for (size_t j0 = 0; j0 < n; j0 += block_size) {
                size_t j_end = (j0 + block_size < n) ? j0 + block_size : n;
                
                for (size_t p0 = 0; p0 < k; p0 += block_size) {
                    size_t p_end = (p0 + block_size < k) ? p0 + block_size : k;
                    
                    // 计算当前块
                    for (size_t i = i0; i < i_end; i++) {
                        for (size_t j = j0; j < j_end; j += 8) {
                            if (j + 8 <= j_end) {
                                // 向量化处理8个元素
                                __m512d sum = _mm512_loadu_pd(&C[i * n + j]);
                                
                                for (size_t p = p0; p < p_end; p++) {
                                    __m512d a_val = _mm512_set1_pd(A[p * m + i]);
                                    __m512d b_vec = _mm512_loadu_pd(&B[p * n + j]);
                                    sum = _mm512_fmadd_pd(a_val, b_vec, sum);
                                }
                                
                                _mm512_storeu_pd(&C[i * n + j], sum);
                            } else {
                                // 处理剩余元素
                                for (size_t jj = j; jj < j_end; jj++) {
                                    double sum = C[i * n + jj];
                                    for (size_t p = p0; p < p_end; p++) {
                                        sum += A[p * m + i] * B[p * n + jj];
                                    }
                                    C[i * n + jj] = sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else { // COL_MAJOR
        // 列主序实现（省略）
    }
}

// 优化版本的TSMM (整合所有优化): C = A^T * B
void tsmm_optimized(const double* A, const double* B, double* C, 
                   size_t m, size_t n, size_t k, MatrixLayout layout) {
    // 根据矩阵尺寸选择最优实现
    if (m == 4000 && n == 16000 && k == 128) {
        // 大矩阵 (4000x16000x128)
        tsmm_large_matrix(A, B, C, m, n, k, layout);
    }
    else if (m == 8 && n == 16 && k == 16000) {
        // 瘦高矩阵1 (8x16x16000)
        tsmm_tall_thin1(A, B, C, m, n, k, layout);
    }
    else if (m == 32 && n == 16000 && k == 16) {
        // 瘦高矩阵2 (32x16000x16)
        tsmm_tall_thin2(A, B, C, m, n, k, layout);
    }
    else if (m == 144 && n == 144 && k == 144) {
        // 方形矩阵 (144x144x144)
        tsmm_square_matrix(A, B, C, m, n, k, layout);
    }
    else {
        // 通用实现 - 使用方形矩阵的逻辑作为兜底
        tsmm_square_matrix(A, B, C, m, n, k, layout);
    }
} 