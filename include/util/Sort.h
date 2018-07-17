#ifndef TRENCHANTTRACER_SORT_H
#define TRENCHANTTRACER_SORT_H

typedef int(*SortCompareFunc)(void *data, int idxA, int idxB);

typedef void(*SortSwapFunc)(void *data, int idxA, int idxB);

void sort(int start, int end, void *data, SortCompareFunc compareFunc, SortSwapFunc swapFunc);

int compareS32(void *data, int idxA, int idxB);

void swapS32(void *data, int idxA, int idxB);

int compareF32(void *data, int idxA, int idxB);

void swapF32(void *data, int idxA, int idxB);


template<class T>
inline void swap(T &a, T &b) {
    T t = a;
    a = b;
    b = t;
}

template<class A, class B>
inline A lerp(const A &a, const A &b, const B &t) {
    return (A) (a * ((B) 1 - t) + b * t);
}


#endif //TRENCHANTTRACER_SORT_H
