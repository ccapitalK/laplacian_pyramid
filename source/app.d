import std.algorithm;
import std.exception;
import std.math;
import std.parallelism;
import std.range;
import std.stdio;

import mir.ndslice;
import simple_image;

struct Dims {
    size_t width, height;
}

float[2] coordsToO1(Dims dims, float x, float y) => [x / dims.width, y / dims.height];
float[2] coordsFromO1(Dims dims, float x, float y) => [x * dims.width, y * dims.height];

Dims dims(in Image im) => Dims(im.width, im.height);
Dims dims(in MirImage im) => Dims(im.length!0, im.length!1);

Image image(in Dims dims) => Image(dims.width, dims.height);
MirImage mirImage(in Dims dims) => slice!float(dims.width, dims.height, 3);

alias MirImage = Slice!(float*, 3, Contiguous);

Slice!(float*, 3, Contiguous) toNdSlice(Image im) {
    // TODO: SRGBify this
    // TODO: Double check the pitch/stride?
    auto dims = im.dims;
    auto matrix = dims.mirImage;
    foreach (y; std.range.iota(0, dims.height).parallel) {
        foreach (x; 0 .. dims.width) {
            // XXX Choose a convention here and stick with it
            matrix[x, y] [] = im.pixel(x, y);
            matrix[x, y] [] /= 255.0;
        }
    }
    return matrix;
}

Image toImage(Slice!(float*, 3, Contiguous) mat) {
    auto dims = mat.dims;
    auto image = dims.image;
    foreach (y; std.range.iota(0, dims.height).parallel) {
        auto pixel = slice!float(3);
        foreach (x; 0 .. dims.width) {
            // XXX Inverse of encode convention
            pixel []= (mat[x, y] * 255.999);
            foreach (i; 0 .. 3) {
                image.pixel(x, y)[i] = cast(ubyte) pixel[i];
            }
        }
    }
    return image;
}

// Sufficient for upscaling, and for downscaling by a factor of at most 2, which is what we are doing
float[3] bilerp(in MirImage mat, float x, float y) {
    float[3] v = [0, 0, 0];
    auto xI = cast(size_t) x.floor;
    auto yI = cast(size_t) y.floor;
    x -= xI;
    y -= yI;
    // FIXME: Optimize this?
    struct Weight {
        size_t index;
        float weight;
    }
    // FIXM: There is probably a more performant way of doing this. We need this to avoid OOB at the edge reads
    Weight[2] xS = [Weight(xI, 1 - x), Weight(x > 0 ? xI + 1 : xI, x)];
    Weight[2] yS = [Weight(yI, 1 - y), Weight(y > 0 ? yI + 1 : yI, y)];
    foreach (xP; xS) {
        foreach (yP; yS) {
            float weight = (0.25 * xP.weight * yP.weight);
            v[] += mat[xP.index, yP.index].array[] * weight;
        }
    }
    return v;
}

// Halve the resolution of the image
MirImage downscale2(in MirImage image) {
    auto dims = image.dims;
    enforce(dims.width > 1 && dims.height > 1);
    auto halfDims = Dims(dims.width / 2, dims.height / 2);
    auto outImage = halfDims.mirImage;
    foreach (y; std.range.iota(0, halfDims.height).parallel) {
        float vy = (y / cast (float) halfDims.height) * dims.height;
        foreach (x; 0 .. halfDims.width) {
            float vx = (x / cast (float) halfDims.width) * dims.width;
            outImage[x, y][] = image.bilerp(vx, vy);
        }
    }
    return outImage;
}

// Need to specify output dimensions since we lose a bit of information downscaling
MirImage upscale2(in MirImage image, Dims outDims) {
    auto dims = image.dims;
    auto outImage = outDims.mirImage;
    foreach (y; std.range.iota(0, outDims.height).parallel) {
        float vy = (y / cast (float) outDims.height) * dims.height;
        foreach (x; 0 .. outDims.width) {
            float vx = (x / cast (float) outDims.width) * dims.width;
            outImage[x, y][] = image.bilerp(vx, vy);
        }
    }
    return outImage;
}

struct LaplacianPyramid {
    MirImage[] pyramid; // Index 0 is the lowest res as a delta from pitch black, the rest are deltas from the previous
}

LaplacianPyramid makePyramid(MirImage image, int levels) {
    auto minSize = 4 << levels;
    enforce(image.dims.width >= minSize && image.dims.height >= minSize);
    LaplacianPyramid pyramid;
    return pyramid;
}

MirImage reconstruct(LaplacianPyramid pyramid) {
    // TODO
}

LaplacianPyramid blend(LaplacianPyramid pyr1, LaplacianPyramid pyr2) {
    enforce(pyr1.pyramid.length == pyr2.pyramid.length);
    return LaplacianPyramid();
}

Image blend(Image im1, Image im2, int nLevels = 6) {
    enforce(im1.width == im2.width && im1.height == im2.height);
    auto pyr1 = im1.makePyramid(nLevels);
    auto pyr2 = im2.makePyramid(nLevels);
    auto blended = blend(im1, im2);
    // return im1;
    return blended.reconstruct;
}

void main(string[] args) {
    enforce(args.length == 3);
    auto im1 = args[1].loadImageRgb;
    auto im2 = args[2].loadImageRgb;
    blend(im1, im2).writeImageRgb("out.bmp");
}
