#ifndef TRENCHANTTRACER_INTERSECTKERNEL_H
#define TRENCHANTTRACER_INTERSECTKERNEL_H

#include <geometry/Ray.h>
#include <geometry/Hit.h>
#include <bvh/BVHCompact.h>

#define STACK_SIZE  64  // Size of the traversal stack in local memory.
#define EntrypointSentinel 0x76543210

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

__device__ __inline__ void swap2(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

__device__ __inline__ void normalize(Hit &hit, const Ray &ray) {
    hit.n = hit.normal;
    hit.n.normalize();
    hit.nl = dot(hit.n, ray.direction) < 0 ? hit.n : hit.n * -1;  // correctly oriented normal
}

__device__ __inline__ void hitPoint(Hit &hit, const Ray &ray) {
    hit.point = ray.origin + ray.direction * hit.distance; // intersection point
}

__device__ __inline__ Hit intersect(const Ray &ray, const BVHCompact *bvhCompact, bool needClosestHit) {
    int traversalStack[STACK_SIZE];

    float idirx, idiry, idirz;    // 1 / dir
    float oodx, oody, oodz;       // orig / dir
    Hit hit;

    char *stackPtr;
    int leafAddr;
    int nodeAddr;

    hit.distance = ray.tMax;

    // ooeps is very small number, used instead of ray direction xyz component when that component is near zero
    float ooeps = exp2f(-80.0f); // Avoid div by zero, returns 1/2^80, an extremely small number
    // inverse ray direction
    idirx = 1.0f / (fabsf(ray.direction.x) > ooeps ? ray.direction.x : copysignf(ooeps, ray.direction.x));
    // inverse ray direction
    idiry = 1.0f / (fabsf(ray.direction.y) > ooeps ? ray.direction.y : copysignf(ooeps, ray.direction.y));
    // inverse ray direction
    idirz = 1.0f / (fabsf(ray.direction.z) > ooeps ? ray.direction.z : copysignf(ooeps, ray.direction.z));
    oodx = ray.origin.x * idirx;  // ray origin / ray direction
    oody = ray.origin.y * idiry;  // ray origin / ray direction
    oodz = ray.origin.z * idirz;  // ray origin / ray direction

    traversalStack[0] = EntrypointSentinel; // Bottom-most entry. 0x76543210 is 1985229328 in decimal
    stackPtr = (char *) &traversalStack[0]; // point stackPtr to bottom of traversal stack = EntryPointSentinel
    leafAddr = 0;   // No postponed leaf.
    nodeAddr = 0;   // Start from the root.

    // EntrypointSentinel = 0x76543210
    while (nodeAddr != EntrypointSentinel) {
        // Traverse internal nodes until all SIMD lanes have found a leaf.

        while (nodeAddr >= 0 && nodeAddr != EntrypointSentinel) {
            float4 n0xy = tex1Dfetch<float4>(bvhCompact->nodesTexture, nodeAddr);
            // child node 0, xy-bounds (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
            float4 n1xy = tex1Dfetch<float4>(bvhCompact->nodesTexture, nodeAddr + 1);
            // child node 1. xy-bounds (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
            float4 nz = tex1Dfetch<float4>(bvhCompact->nodesTexture, nodeAddr + 2);
            // child nodes 0 and 1, z-bounds(c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
            float4 tmp = tex1Dfetch<float4>(bvhCompact->nodesTexture, nodeAddr + 3);
            // contains indices to 2 child nodes in case of inner node, see below

            // ptr[3] contains indices to 2 child nodes in case of inner node, see below
            // (child index = size of array during building, see CudaBVH.cpp)

            // compute ray intersections with BVH node bounding box
            float c0lox = n0xy.x * idirx - oodx; // n0xy.x = c0.lo.x, child 0 min bound x
            float c0hix = n0xy.y * idirx - oodx; // n0xy.y = c0.hi.x, child 0 max bound x
            float c0loy = n0xy.z * idiry - oody; // n0xy.z = c0.lo.y, child 0 min bound y
            float c0hiy = n0xy.w * idiry - oody; // n0xy.w = c0.hi.y, child 0 max bound y
            float c0loz = nz.x * idirz - oodz; // nz.x   = c0.lo.z, child 0 min bound z
            float c0hiz = nz.y * idirz - oodz; // nz.y   = c0.hi.z, child 0 max bound z
            float c1loz = nz.z * idirz - oodz; // nz.z   = c1.lo.z, child 1 min bound z
            float c1hiz = nz.w * idirz - oodz; // nz.w   = c1.hi.z, child 1 max bound z
            float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, ray.tMin);
            // Tesla does max4(min, min, min, tmin)
            float c0max = spanEndKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hit.distance);
            // Tesla does min4(max, max, max, tmax)
            float c1lox = n1xy.x * idirx - oodx; // n1xy.x = c1.lo.x, child 1 min bound x
            float c1hix = n1xy.y * idirx - oodx; // n1xy.y = c1.hi.x, child 1 max bound x
            float c1loy = n1xy.z * idiry - oody; // n1xy.z = c1.lo.y, child 1 min bound y
            float c1hiy = n1xy.w * idiry - oody; // n1xy.w = c1.hi.y, child 1 max bound y
            float c1min = spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, ray.tMin);
            float c1max = spanEndKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, hit.distance);

            float ray_tmax = 1e20;
            bool traverseChild0 = (c0min <= c0max) && (c0min >= ray.tMin) && (c0min <= ray_tmax);
            bool traverseChild1 = (c1min <= c1max) && (c1min >= ray.tMin) && (c1min <= ray_tmax);

            if (!traverseChild0 && !traverseChild1) {
                nodeAddr = *(int *) stackPtr; // fetch next node by popping stack
                stackPtr -= 4; // popping decrements stack by 4 bytes (because stackPtr is a pointer to char)
            } else {
                // Otherwise => fetch child pointers.
                // one or both children intersected
                int2 childNodes = *(int2 *) &tmp; // cast first two floats to int
                // set nodeAddr equal to intersected childnode (first childnode when both children are intersected)
                nodeAddr = (traverseChild0) ? childNodes.x : childNodes.y;

                // Both children were intersected => push the farther one on the stack.
                if (traverseChild0 && traverseChild1) {
                    // store closest child in nodeAddr, swap if necessary
                    if (c1min < c0min)
                        swap2(nodeAddr, childNodes.y);
                    stackPtr += 4;  // pushing increments stack by 4 bytes (stackPtr is a pointer to char)
                    *(int *) stackPtr = childNodes.y; // push furthest node on the stack
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

            // Intersect the ray against each triangle using Sven Woop's algorithm.
            // Woop ray triangle intersection: Woop triangles are unit triangles. Each ray
            // must be transformed to "unit triangle space", before testing for intersection
            for (int triAddr = ~leafAddr;; triAddr += 3) {
                // no defined upper limit for loop, continues until leaf terminator code 0x80000000 is encountered

                // Read first 16 bytes of the triangle.
                // fetch first precomputed triangle edge
                float4 v00 = tex1Dfetch<float4>(bvhCompact->woopTriTexture, triAddr);

                // End marker 0x80000000 (= negative zero) => all triangles in leaf processed. --> terminate
                if (__float_as_int(v00.x) == 0x80000000) break;

                // Compute and check intersection t-value (hit distance along ray).
                // Origin z
                float Oz = v00.w - ray.origin.x * v00.x - ray.origin.y * v00.y - ray.origin.z * v00.z;
                // inverse Direction z
                float invDz = 1.0f / (ray.direction.x * v00.x + ray.direction.y * v00.y + ray.direction.z * v00.z);
                float t = Oz * invDz;

                if (ray.tMin < t && t < hit.distance) {
                    // Compute and check barycentric u.

                    // fetch second precomputed triangle edge
                    float4 v11 = tex1Dfetch<float4>(bvhCompact->woopTriTexture, triAddr + 1);
                    float Ox = v11.w + ray.origin.x * v11.x + ray.origin.y * v11.y + ray.origin.z * v11.z;  // Origin.x
                    float Dx =
                            ray.direction.x * v11.x + ray.direction.y * v11.y + ray.direction.z * v11.z;  // Direction.x
                    float u = Ox + t * Dx; // parametric equation of a ray (intersection point)

                    if (u >= 0.0f && u <= 1.0f) {
                        // Compute and check barycentric v.
                        // fetch third precomputed triangle edge
                        float4 v22 = tex1Dfetch<float4>(bvhCompact->woopTriTexture, triAddr + 2);
                        float Oy = v22.w + ray.origin.x * v22.x + ray.origin.y * v22.y + ray.origin.z * v22.z;
                        float Dy = ray.direction.x * v22.x + ray.direction.y * v22.y + ray.direction.z * v22.z;
                        float v = Oy + t * Dy;

                        if (v >= 0.0f && u + v <= 1.0f) {
                            // We've got a hit!
                            // Record intersection.
                            hit.distance = t;
                            hit.index = triAddr; // store triangle index for shading

                            Vec3f n0 = Vec3f(tex1Dfetch<float4>(bvhCompact->normalsTexture, triAddr));
                            Vec3f n1 = Vec3f(tex1Dfetch<float4>(bvhCompact->normalsTexture, triAddr + 1));
                            Vec3f n2 = Vec3f(tex1Dfetch<float4>(bvhCompact->normalsTexture, triAddr + 2));
                            // Interpolate to find normal
                            hit.normal = n0 * (1 - u - v) + n1 * u + n2 * v;

                            if (!needClosestHit) {
                                // shadow rays only require "any" hit with scene geometry, not the closest one
                                nodeAddr = EntrypointSentinel;
                                break;
                            }
                        }
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
        hit.matIndex = tex1Dfetch<int>(bvhCompact->matIndicesTexture, hit.index);
        hit.index = tex1Dfetch<int>(bvhCompact->triIndicesTexture, hit.index);

//        normalize(hit, ray);
//        hitPoint(hit, ray);
    }

    return hit;
}

#endif //TRENCHANTTRACER_INTERSECTKERNEL_H
