#include <loader/OBJLoader.h>

Object *OBJLoader::load(std::string fileName) {
    std::cout << "loading OBJ model: " << fileName << std::endl;
    std::ifstream ifs(fileName);
    if (!ifs.good())
        throw std::runtime_error("ERROR: loading obj, file not found or not good");

    auto *object = new Object();

    MaterialLoader materialLoader;

    Mesh *mesh = nullptr;
    Group *group = nullptr;
    MaterialPool *materialPool = nullptr;
    std::string line, key;
    while (!ifs.eof() && std::getline(ifs, line)) {
        key = "";
        std::stringstream lineStream(line);
        lineStream >> key >> std::ws;

        if (key == "g") { // group
            std::string name;
            lineStream >> name;
            mesh = new Mesh();
            group = new Group(name, mesh);
            object->addGroup(group);
        } else if (key == "mtllib") {
            std::string name;
            std::string prefix;
            lineStream >> name;
            unsigned long index = fileName.find_last_of('/');
            if (index != std::string::npos)
                prefix = fileName.substr(0, index + 1);
            materialPool = materialLoader.load(prefix + name);
            object->setMaterialPool(materialPool);
        } else if (key == "usemtl") {
            std::string name;
            lineStream >> name;
            Material *material = materialPool->get(name);

            if (material == nullptr)
                throw std::runtime_error("Material " + name + " not found");
            group->setMaterial(material);
        } else if (key == "v") { // vertex
            float x, y, z;
            while (!lineStream.eof()) {
                lineStream >> x >> std::ws >> y >> std::ws >> z >> std::ws;
                object->addVertex(Vec3f(x, y, z));
            }
        } else if (key == "vp") { // parameter
            float x;
            while (!lineStream.eof()) {
                lineStream >> x >> std::ws;
            }
        } else if (key == "vt") { // texture coordinate
            float x;
            while (!lineStream.eof()) {
                lineStream >> x >> std::ws;
            }
        } else if (key == "vn") { // normal
            float x;
            while (!lineStream.eof()) {
                lineStream >> x >> std::ws;
            }
        } else if (key == "f") { // face
            int v = 1, t = 1, n = 1;
            std::vector<int> vertices;
            std::vector<int> textures;
            std::vector<int> normals;
            while (!lineStream.eof()) {
                lineStream >> v >> std::ws;
                if (lineStream.peek() == '/') {
                    lineStream.get();
                    if (lineStream.peek() == '/') {
                        lineStream >> n >> std::ws;
                    } else {
                        lineStream >> t >> std::ws;
                        if (lineStream.peek() == '/') {
                            lineStream.get();
                            lineStream >> n >> std::ws;
                        }
                    }
                }

                vertices.push_back(v - 1);
                textures.push_back(t - 1);
                normals.push_back(n - 1);
            }

            auto numTriangles = static_cast<int>(vertices.size() - 2); // 1 triangle if 3 vertices, 2 if 4 etc
            for (int i = 0; i < numTriangles; i++) {
                // first vertex remains the same for all triangles in a triangle fan
                mesh->addVertex(Vec3i(vertices[0], vertices[i + 1], vertices[i + 2]));
                mesh->addTexture(Vec3i(textures[0], textures[i + 1], textures[i + 2]));
                mesh->addNormal(Vec3i(normals[0], normals[i + 1], normals[i + 2]));
            }
        }
    }

    ifs.close();

    object->postProcess();
    return object;
}
