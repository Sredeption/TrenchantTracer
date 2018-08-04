#ifndef TRENCHANTTRACER_MATERIALUNION_H
#define TRENCHANTTRACER_MATERIALUNION_H

#include <material/Coat.h>
#include <material/Diff.h>
#include <material/Metal.h>
#include <material/Spec.h>
#include <material/Refr.h>

union MaterialUnion {
    Coat coat;
    Diff diff;
    Metal metal;
    Spec spec;
    Refr refr;
};

#endif //TRENCHANTTRACER_MATERIALUNION_H
