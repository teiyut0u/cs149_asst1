
#include <math.h>
#include <stdlib.h>
#include <x86intrin.h>

static const __m256 __m256_one_vec = _mm256_set1_ps(1.f);
static const __m256 __m256_half_vec = _mm256_set1_ps(.5f);
static const __m256 __m256_3_vec = _mm256_set1_ps(3.f);
static const __m256 __m256_kThershold_vec = _mm256_set1_ps(0.00001f);
static const __m256 __m256_abs_mask =
    _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

void sqrtSerial(int N, float initialGuess, float values[], float output[]) {

  static const float kThreshold = 0.00001f;

  for (int i = 0; i < N; i++) {

    float x = values[i];
    float guess = initialGuess;

    float error = fabs(guess * guess * x - 1.f);

    while (error > kThreshold) {
      guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
      error = fabs(guess * guess * x - 1.f);
    }

    output[i] = x * guess;
  }
}

void sqrtSerialAVX2(int N, float initialGuess, float values[], float output[]) {

  int end = N & ~(7);
  int i = 0;

work:
  for (; i < end; i += 8) {
    __m256 x = _mm256_loadu_ps(values + i);
    __m256 guess = _mm256_set1_ps(initialGuess);

    __m256 error = _mm256_and_ps(
        _mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(guess, guess), x),
                      __m256_one_vec),
        __m256_abs_mask);

    __m256 gt_mask = _mm256_cmp_ps(error, __m256_kThershold_vec, _CMP_GT_OQ);

    while (_mm256_movemask_ps(gt_mask)) {

      guess = _mm256_blendv_ps(
          guess,
          _mm256_mul_ps(
              _mm256_mul_ps(
                  _mm256_sub_ps(__m256_3_vec,
                                _mm256_mul_ps(_mm256_mul_ps(x, guess), guess)),
                  guess),
              __m256_half_vec),
          gt_mask);

      error = _mm256_and_ps(
          _mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(guess, guess), x),
                        __m256_one_vec),
          __m256_abs_mask);

      gt_mask = _mm256_cmp_ps(error, __m256_kThershold_vec, _CMP_GT_OQ);
    }
    _mm256_storeu_ps(output + i, _mm256_mul_ps(x, guess));
  }
  if (N > end) {
    i = N - 8;
    end = N;
    goto work;
  }
}

void sqrtSerialAVX2Aligned(int N, float initialGuess, float values[],
                           float output[]) {

  // float *start_p = reinterpret_cast<float *>(
  //           (reinterpret_cast<uintptr_t>(values) + 31) & (~31)),
  //       *end_p = reinterpret_cast<float *>(
  //           reinterpret_cast<uintptr_t>(values + N) & (~31));

  int end = N & ~(7);
  int i = 0;

work:
  for (; i < end; i += 8) {
    __m256 x = _mm256_loadu_ps(values + i);
    __m256 guess = _mm256_set1_ps(initialGuess);

    __m256 error = _mm256_and_ps(
        _mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(guess, guess), x),
                      __m256_one_vec),
        __m256_abs_mask);

    __m256 gt_mask = _mm256_cmp_ps(error, __m256_kThershold_vec, _CMP_GT_OQ);

    while (_mm256_movemask_ps(gt_mask)) {

      guess = _mm256_blendv_ps(
          guess,
          _mm256_mul_ps(
              _mm256_mul_ps(
                  _mm256_sub_ps(__m256_3_vec,
                                _mm256_mul_ps(_mm256_mul_ps(x, guess), guess)),
                  guess),
              __m256_half_vec),
          gt_mask);

      error = _mm256_and_ps(
          _mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(guess, guess), x),
                        __m256_one_vec),
          __m256_abs_mask);

      gt_mask = _mm256_cmp_ps(error, __m256_kThershold_vec, _CMP_GT_OQ);
    }
    _mm256_storeu_ps(output + i, _mm256_mul_ps(x, guess));
  }
  if (N > end) {
    i = N - 8;
    end = N;
    goto work;
  }
}