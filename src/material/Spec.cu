#include <material/Spec.h>

const std::string Spec::TYPE = "Spec";

__host__ __device__ Spec::Spec() : Material(SPEC) {

}

__host__ Spec::Spec(const nlohmann::json &material) : Spec() {
    color = jsonToColor(material["color"]);
}

__host__ U32 Spec::size() const {
    return sizeof(Spec);
}

__device__ Ray Spec::sample(curandState *randState, const Ray &ray, const Hit &hit, Vec3f &mask) {
    Ray nextRay = ray;
    // ray of next path segment
    // compute reflected ray direction according to Snell's law
    nextRay.direction = ray.direction - hit.n * dot(hit.n, ray.direction) * 2.0f;
    nextRay.direction.normalize();

    // offset origin next path segment to prevent self intersection
    nextRay.origin = hit.point + hit.nl * 0.001f;

    // multiply mask with colour of object
    mask *= color;
    return nextRay;
}
