#include <geometry/Ray.h>
#include <cuda_runtime.h>

//  RAY BOX INTERSECTION ROUTINES

// Experimentally determined best mix of float/int/video minmax instructions for Kepler.

// float c0min = spanBeginKepler2(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin); // Tesla does max4(min, min, min, tmin)
// float c0max = spanEndKepler2(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT); // Tesla does min4(max, max, max, tmax)

// Perform min/max operations in hardware
// Using Kepler's video instructions, see http://docs.nvidia.com/cuda/parallel-thread-execution/#axzz3jbhbcTZf																			//  : "=r"(v) overwrites v and puts it in a register
// see https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html

__device__ __inline__ int min_min(int a, int b, int c) {
    int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v;
}

__device__ __inline__ int min_max(int a, int b, int c) {
    int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v;
}

__device__ __inline__ int max_min(int a, int b, int c) {
    int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v;
}

__device__ __inline__ int max_max(int a, int b, int c) {
    int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v;
}

__device__ __inline__ float fmin_fmin(float a, float b, float c) {
    return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c)));
}

__device__ __inline__ float fmin_fmax(float a, float b, float c) {
    return __int_as_float(min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c)));
}

__device__ __inline__ float fmax_fmin(float a, float b, float c) {
    return __int_as_float(max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c)));
}

__device__ __inline__ float fmax_fmax(float a, float b, float c) {
    return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c)));
}

__device__ __inline__ float spanBeginKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) {
    return fmax_fmax(fminf(a0, a1), fminf(b0, b1), fmin_fmax(c0, c1, d));
}

__device__ __inline__ float spanEndKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) {
    return fmin_fmin(fmaxf(a0, a1), fmaxf(b0, b1), fmax_fmin(c0, c1, d));
}

// standard ray box intersection routines (for debugging purposes only)
// based on Intersect::RayBox() in original Aila/Laine code
__device__ __inline__ float
spanBeginKepler2(float lo_x, float hi_x, float lo_y, float hi_y, float lo_z, float hi_z, float d) {

    Vec3f t0 = Vec3f(lo_x, lo_y, lo_z);
    Vec3f t1 = Vec3f(hi_x, hi_y, hi_z);

    Vec3f realmin = min3f(t0, t1);

    float raybox_tmin = realmin.max(); // maxmin

    //return Vec2f(tmin, tmax);
    return raybox_tmin;
}

__device__ __inline__ float
spanEndKepler2(float lo_x, float hi_x, float lo_y, float hi_y, float lo_z, float hi_z, float d) {

    Vec3f t0 = Vec3f(lo_x, lo_y, lo_z);
    Vec3f t1 = Vec3f(hi_x, hi_y, hi_z);

    Vec3f realmax = max3f(t0, t1);

    float raybox_tmax = realmax.min(); /// minmax

    //return Vec2f(tmin, tmax);
    return raybox_tmax;
}

__device__ __inline__ void swap2(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

__device__ Hit Ray::intersect(const BVHCompact *bvh, bool needClosestHit) {
    int traversalStack[STACK_SIZE];

    float idirx, idiry, idirz;    // 1 / dir
    float oodx, oody, oodz;       // orig / dir
    Hit hit;

    char *stackPtr;
    int leafAddr;
    int nodeAddr;

    hit.distance = tMax;

    // ooeps is very small number, used instead of ray direction xyz component when that component is near zero
    float ooeps = exp2f(-80.0f); // Avoid div by zero, returns 1/2^80, an extremely small number
    idirx = 1.0f / (fabsf(direction.x) > ooeps ? direction.x : copysignf(ooeps, direction.x)); // inverse ray direction
    idiry = 1.0f / (fabsf(direction.y) > ooeps ? direction.y : copysignf(ooeps, direction.y)); // inverse ray direction
    idirz = 1.0f / (fabsf(direction.z) > ooeps ? direction.z : copysignf(ooeps, direction.z)); // inverse ray direction
    oodx = origin.x * idirx;  // ray origin / ray direction
    oody = origin.y * idiry;  // ray origin / ray direction
    oodz = origin.z * idirz;  // ray origin / ray direction

    traversalStack[0] = EntrypointSentinel; // Bottom-most entry. 0x76543210 is 1985229328 in decimal
    stackPtr = (char *) &traversalStack[0]; // point stackPtr to bottom of traversal stack = EntryPointSentinel
    leafAddr = 0;   // No postponed leaf.
    nodeAddr = 0;   // Start from the root.

    // EntrypointSentinel = 0x76543210
    while (nodeAddr != EntrypointSentinel) {
        // Traverse internal nodes until all SIMD lanes have found a leaf.

        while (nodeAddr >= 0 && nodeAddr != EntrypointSentinel) {
            float4 *ptr = bvh->nodes + nodeAddr;
            float4 n0xy = ptr[0]; // childnode 0, xy-bounds (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
            float4 n1xy = ptr[1]; // childnode 1. xy-bounds (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
            float4 nz = ptr[2]; // childnodes 0 and 1, z-bounds(c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)

            // ptr[3] contains indices to 2 childnodes in case of innernode, see below
            // (childindex = size of array during building, see CudaBVH.cpp)

            // compute ray intersections with BVH node bounding box
            float c0lox = n0xy.x * idirx - oodx; // n0xy.x = c0.lo.x, child 0 min bound x
            float c0hix = n0xy.y * idirx - oodx; // n0xy.y = c0.hi.x, child 0 max bound x
            float c0loy = n0xy.z * idiry - oody; // n0xy.z = c0.lo.y, child 0 min bound y
            float c0hiy = n0xy.w * idiry - oody; // n0xy.w = c0.hi.y, child 0 max bound y
            float c0loz = nz.x * idirz - oodz; // nz.x   = c0.lo.z, child 0 min bound z
            float c0hiz = nz.y * idirz - oodz; // nz.y   = c0.hi.z, child 0 max bound z
            float c1loz = nz.z * idirz - oodz; // nz.z   = c1.lo.z, child 1 min bound z
            float c1hiz = nz.w * idirz - oodz; // nz.w   = c1.hi.z, child 1 max bound z
            float c0min = spanBeginKepler2(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tMin);
            // Tesla does max4(min, min, min, tmin)
            float c0max = spanEndKepler2(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hit.distance);
            // Tesla does min4(max, max, max, tmax)
            float c1lox = n1xy.x * idirx - oodx; // n1xy.x = c1.lo.x, child 1 min bound x
            float c1hix = n1xy.y * idirx - oodx; // n1xy.y = c1.hi.x, child 1 max bound x
            float c1loy = n1xy.z * idiry - oody; // n1xy.z = c1.lo.y, child 1 min bound y
            float c1hiy = n1xy.w * idiry - oody; // n1xy.w = c1.hi.y, child 1 max bound y
            float c1min = spanBeginKepler2(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, tMin);
            float c1max = spanEndKepler2(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, hit.distance);

            float ray_tmax = 1e20;
            bool traverseChild0 = (c0min <= c0max) && (c0min >= tMin) && (c0min <= ray_tmax);
            bool traverseChild1 = (c1min <= c1max) && (c1min >= tMin) && (c1min <= ray_tmax);

            if (!traverseChild0 && !traverseChild1) {
                nodeAddr = *(int *) stackPtr; // fetch next node by popping stack
                stackPtr -= 4; // popping decrements stack by 4 bytes (because stackPtr is a pointer to char)
            } else {
                // Otherwise => fetch child pointers.
                // one or both children intersected
                int2 cnodes = *(int2 *) &ptr[3];
                // set nodeAddr equal to intersected childnode (first childnode when both children are intersected)
                nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;

                // Both children were intersected => push the farther one on the stack.
                if (traverseChild0 && traverseChild1) {
                    // store closest child in nodeAddr, swap if necessary
                    if (c1min < c0min)
                        swap2(nodeAddr, cnodes.y);
                    stackPtr += 4;  // pushing increments stack by 4 bytes (stackPtr is a pointer to char)
                    *(int *) stackPtr = cnodes.y; // push furthest node on the stack
                }
            }

            // First leaf => postpone and continue traversal.
            // leaf nodes have a negative index to distinguish them from inner nodes
            // if nodeAddr less than 0 -> nodeAddr is a leaf
            if (nodeAddr < 0 && leafAddr >= 0) {
                // if leafAddr >= 0 -> no leaf found yet (first leaf)
                leafAddr = nodeAddr;

                nodeAddr = *(int *) stackPtr;  // pops next node from stack
                stackPtr -= 4;  // decrement by 4 bytes (stackPtr is a pointer to char)
            }

            // All SIMD lanes have found a leaf => process them.
            // NOTE: inline PTX implementation of "if(!__any(leafAddr >= 0)) break;".
            // tried everything with CUDA 4.2 but always got several redundant instructions.

            // if (!searchingLeaf){ break;  }

            // if (!__any(searchingLeaf)) break; // "__any" keyword: if none of the threads is searching a leaf, in other words
            // if all threads in the warp found a leafnode, then break from while loop and go to triangle intersection

            // if(!__any(leafAddr >= 0))   /// als leafAddr in PTX code >= 0, dan is het geen echt leafNode
            //    break;

            unsigned int mask; // mask replaces searchingLeaf in PTX code

            asm("{\n"
                "   .reg .pred p;               \n"
                "setp.ge.s32        p, %1, 0;   \n"
                "vote.ballot.b32    %0,p;       \n"
                "}"
            : "=r"(mask)
            : "r"(leafAddr));

            if (!mask)
                break;
        }

        ///////////////////////////////////////
        /// LEAF NODE / TRIANGLE INTERSECTION
        ///////////////////////////////////////
        while (leafAddr < 0) {
            // if leafAddr is negative, it points to an actual leafnode (when positive or 0 it's an innernode
            // leafAddr is stored as negative number, see cidx[i] = ~triWoopData.getSize(); in CudaBVH.cpp
            int leafCount = 0;
            for (int triAddr = ~leafAddr;; triAddr += 3) {
                leafCount++;
                // no defined upper limit for loop, continues until leaf terminator code 0x80000000 is encountered

                // Read first 16 bytes of the triangle.
                // fetch first triangle vertex
                float4 v0f = bvh->debugTri[triAddr + 0];

                // End marker 0x80000000 (= negative zero) => all triangles in leaf processed. --> terminate
                if (__float_as_int(v0f.x) == 0x80000000) break;

                float4 v1f = bvh->debugTri[triAddr + 1];
                float4 v2f = bvh->debugTri[triAddr + 2];

                const Vec3f v0 = Vec3f(v0f.x, v0f.y, v0f.z);
                const Vec3f v1 = Vec3f(v1f.x, v1f.y, v1f.z);
                const Vec3f v2 = Vec3f(v2f.x, v2f.y, v2f.z);

                // convert float4 to Vec4f
                Vec3f bary = intersect(v0, v1, v2);

                float t = bary.z; // hit distance along ray
                if (tMin < t && t < hit.distance) {
                    // if there is a miss, t will be larger than hitT (ray.tmax)
                    hit.index = triAddr;
                    hit.distance = t;  // keeps track of closest hitpoint
                    hit.noraml = cross(v0 - v1, v0 - v2);

                    if (!needClosestHit) {
                        // shadow rays only require "any" hit with scene geometry, not the closest one
                        nodeAddr = EntrypointSentinel;
                        break;
                    }
                }
            } // triangle

            // Another leaf was postponed => process it as well.

            leafAddr = nodeAddr;

            if (nodeAddr < 0) {
                nodeAddr = *(int *) stackPtr;  // pop stack
                stackPtr -= 4;               // decrement with 4 bytes to get the next int (stackPtr is char*)
            }
        } // end leaf/triangle intersection loop
    } // end of node traversal loop

    // Remap intersected triangle index, and store the result.
    if (hit.index != -1) {
        // remapping tri indices delayed until this point for performance reasons
        // (slow global memory lookup in de gpuTriIndices array) because multiple triangles per node can potentially be hit
        hit.index = bvh->triIndices[hit.index].x;
    }

    return hit;
}

__device__ Vec3f Ray::intersect(const Vec3f &v0, const Vec3f &v1, const Vec3f &v2) {
    const Vec3f rayorig3f = Vec3f(origin.x, origin.y, origin.z);
    const Vec3f raydir3f = Vec3f(direction.x, direction.y, direction.z);

    const float EPSILON = 0.00001f; // works better
    const Vec3f miss(F32_MAX, F32_MAX, F32_MAX);

    Vec3f edge1 = v1 - v0;
    Vec3f edge2 = v2 - v0;

    Vec3f tvec = rayorig3f - v0;
    Vec3f pvec = cross(raydir3f, edge2);
    float det = dot(edge1, pvec);

    float invdet = 1.0f / det;

    float u = dot(tvec, pvec) * invdet;

    Vec3f qvec = cross(tvec, edge1);

    float v = dot(raydir3f, qvec) * invdet;

    if (det > EPSILON) {
        if (u < 0.0f || u > 1.0f) return miss; // 1.0 want = det * 1/det
        if (v < 0.0f || (u + v) > 1.0f) return miss;
        // if u and v are within these bounds, continue and go to float t = dot(...
    } else if (det < -EPSILON) {
        if (u > 0.0f || u < 1.0f) return miss;
        if (v > 0.0f || (u + v) < 1.0f) return miss;
        // else continue
    } else // if det is not larger (more positive) than EPSILON or not smaller (more negative) than -EPSILON, there is a "miss"
        return miss;

    float t = dot(edge2, qvec) * invdet;

    if (t > tMin && t < tMax)
        return Vec3f(u, v, t);

    // otherwise (t < raytmin or t > raytmax) miss
    return miss;
}

