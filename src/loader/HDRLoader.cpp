#include <loader/HDRLoader.h>

HDRLoader::HDRLoader() = default;

HDRImage *HDRLoader::load(std::string fileName) {
    int i;
    char str[200];
    FILE *file;

    file = fopen(fileName.c_str(), "rb");
    if (!file)
        throw std::runtime_error("HDR environment map not found. Exiting now...\n");

    fread(str, 10, 1, file);
    if (memcmp(str, "#?RADIANCE", 10) != 0) {  // check RADIANCE renderer format
        fclose(file);
        throw std::runtime_error("Renderer format is not RADIANCE\n");
    }

    fseek(file, 1, SEEK_CUR);
    char cmd[200];
    i = 0;
    char c = 0, oldc;
    while (true) {
        oldc = c;
        c = static_cast<char>(fgetc(file));
        if (c == 0xa && oldc == 0xa)
            break;
        cmd[i++] = c;
    }

    char reso[200];
    i = 0;
    while (true) {
        c = static_cast<char>(fgetc(file));
        reso[i++] = c;
        if (c == 0xa)
            break;
    }

    long w, h;
    if (!sscanf(reso, "-Y %ld +X %ld", &h, &w)) {
        fclose(file);
        throw std::runtime_error("Width and height format error\n");
    }


    auto *colorsPointer = new float[w * h * 3];
    auto *colors = colorsPointer;
    auto *scanLine = new RGBE[w];

    // convert image
    for (auto y = static_cast<int>(h - 1); y >= 0; y--) {
        if (!decrunch(scanLine, w, file))
            break;
        workOnRGBE(scanLine, w, colorsPointer);
        colorsPointer += w * 3;
    }

    delete[] scanLine;
    fclose(file);
    auto image = new HDRImage(w, h, colors);
    delete[] colors;
    return image;
}

float HDRLoader::convertComponent(int expo, int val) {
    float v = val / 256.0f;
    float d = powf(2, expo);
    return v * d;
}

void HDRLoader::workOnRGBE(RGBE *scan, int len, float *cols) {
    while (len-- > 0) {
        int expo = scan[0][E] - 128;
        cols[0] = convertComponent(expo, scan[0][R]);
        cols[1] = convertComponent(expo, scan[0][G]);
        cols[2] = convertComponent(expo, scan[0][B]);
        cols += 3;
        scan++;
    }
}

bool HDRLoader::decrunch(RGBE *scanLine, int len, FILE *file) {
    int i, j;

    if (len < MIN_LEN || len > MAX_LEN)
        return oldDecrunch(scanLine, len, file);

    i = fgetc(file);
    if (i != 2) {
        fseek(file, -1, SEEK_CUR);
        return oldDecrunch(scanLine, len, file);
    }

    scanLine[0][G] = static_cast<unsigned char>(fgetc(file));
    scanLine[0][B] = static_cast<unsigned char>(fgetc(file));
    i = fgetc(file);

    if (scanLine[0][G] != 2 || scanLine[0][B] & 128) {
        scanLine[0][R] = 2;
        scanLine[0][E] = static_cast<unsigned char>(i);
        return oldDecrunch(scanLine + 1, len - 1, file);
    }

    // read each component
    for (i = 0; i < 4; i++) {
        for (j = 0; j < len;) {
            auto code = static_cast<unsigned char>(fgetc(file));
            if (code > 128) { // run
                code &= 127;
                auto val = static_cast<unsigned char>(fgetc(file));
                while (code--)
                    scanLine[j++][i] = val;
            } else {    // non-run
                while (code--)
                    scanLine[j++][i] = static_cast<unsigned char>(fgetc(file));
            }
        }
    }

    return feof(file) == 0;
}

bool HDRLoader::oldDecrunch(RGBE *scanLine, int len, FILE *file) {
    int i;
    int rshift = 0;

    while (len > 0) {
        scanLine[0][R] = static_cast<unsigned char>(fgetc(file));
        scanLine[0][G] = static_cast<unsigned char>(fgetc(file));
        scanLine[0][B] = static_cast<unsigned char>(fgetc(file));
        scanLine[0][E] = static_cast<unsigned char>(fgetc(file));
        if (feof(file))
            return false;

        if (scanLine[0][R] == 1 &&
            scanLine[0][G] == 1 &&
            scanLine[0][B] == 1) {
            for (i = scanLine[0][E] << rshift; i > 0; i--) {
                memcpy(&scanLine[0][0], &scanLine[-1][0], 4);
                scanLine++;
                len--;
            }
            rshift += 8;
        } else {
            scanLine++;
            len--;
            rshift = 0;
        }
    }
    return true;
}
