import std.algorithm;
import std.exception;
import std.functional : toDelegate;
import std.parallelism;
import std.math;
import std.range;
import std.stdio;

import mir.ndslice;
import simple_image;

struct Dims {
    size_t width, height;
}

float[2] coordsToO1(float[2] c, Dims dims) => [c[0] / dims.width, c[1] / dims.height];
float[2] coordsFromO1(float[2] c, Dims dims) => [c[0] * dims.width, c[1] * dims.height];

float[2] vec2(size_t x, size_t y) => [cast(float) x, cast(float) y];
float[2] vec2(float x, float y) => [x, y];

Dims dims(in ref Image im) => Dims(im.width, im.height);
Dims dims(in ref MirImage im) => Dims(im.length!0, im.length!1);

Image image(in Dims dims) => Image(dims.width, dims.height);
MirImage mirImage(in Dims dims) => slice!float(dims.width, dims.height, 3);

size_t width(in ref MirImage im) => im.length!0;
size_t height(in ref MirImage im) => im.length!1;

alias MirImage = Slice!(float*, 3, Contiguous);

Slice!(float*, 3, Contiguous) toMirImage(Image im) {
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
            pixel []= (mat[x, y] * 255);
            foreach (i; 0 .. 3) {
                image.pixel(x, y)[i] = cast(ubyte) clamp(pixel[i], 0f, 255f);
            }
        }
    }
    return image;
}

// Sufficient for upscaling, and for downscaling by a factor of at most 2, which is what we are doing
// `x` should be in [0, mat.width), and `y` in [0, mat.height)
float[3] bilerp(in MirImage mat, float x, float y) {
    auto dims = mat.dims;
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
    // FIXME: This is actually subtly off, need to rethink what pixels as samples would mean here. Range being [) has
    // introduced error here,
    // FIXME: There is probably a more performant way of doing the clamp. We need this to avoid OOB at the edge reads
    Weight[2] xS = [Weight(xI, 1 - x), Weight(xI + 1 < dims.width ? xI + 1 : xI, x)];
    Weight[2] yS = [Weight(yI, 1 - y), Weight(yI + 1 < dims.height ? yI + 1 : yI, y)];
    foreach (xP; xS) {
        foreach (yP; yS) {
            float weight = xP.weight * yP.weight;
            if (!(xP.index < dims.width && yP.index < dims.height)) {
                writefln!"%s %s %s"(dims, xP, yP);
            }
            enforce(xP.index < dims.width && yP.index < dims.height);
            float[3] texel;
            foreach (i; 0 .. 3) {
                texel[i] = mat[xP.index, yP.index, i];
            }
            v[] += texel[] * weight;
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
        foreach (x; 0 .. halfDims.width) {
            auto c = vec2(x, y).coordsToO1(halfDims).coordsFromO1(dims);
            outImage[x, y][] = image.bilerp(c[0], c[1]);
        }
    }
    return outImage;
}

// Need to specify output dimensions since we lose a bit of information downscaling
MirImage upscale2(in MirImage image, Dims outDims) {
    auto dims = image.dims;
    auto outImage = outDims.mirImage;
    foreach (y; std.range.iota(0, outDims.height).parallel) {
        foreach (x; 0 .. outDims.width) {
            auto c = vec2(x, y).coordsToO1(outDims).coordsFromO1(dims);
            outImage[x, y][] = image.bilerp(c[0], c[1]);
        }
    }
    return outImage;
}

MirImage add(MirImage a, MirImage b) {
    enforce(a.dims == b.dims);
    auto dims = a.dims;
    auto outImage = dims.mirImage;
    foreach (y; std.range.iota(0, dims.height).parallel) {
        foreach (x; 0 .. dims.width) {
            outImage[x, y][] = a[x, y] + b[x, y];
        }
    }
    return outImage;
}

MirImage subtract(MirImage base, MirImage valToSubtract) {
    enforce(base.dims == valToSubtract.dims);
    auto dims = base.dims;
    auto outImage = dims.mirImage;
    foreach (y; std.range.iota(0, dims.height).parallel) {
        foreach (x; 0 .. dims.width) {
            outImage[x, y][] = base[x, y] - valToSubtract[x, y];
        }
    }
    return outImage;
}

alias BlendFunc = float delegate(float x, float y);

MirImage performBlend(MirImage a, MirImage b, BlendFunc f) {
    enforce(a.dims == b.dims);
    auto dims = a.dims;
    auto outImage = dims.mirImage;
    foreach (y; std.range.iota(0, dims.height).parallel) {
        foreach (x; 0 .. dims.width) {
            auto b1 = f(x / cast(float) dims.width, y / cast(float) dims.height);
            auto b0 = 1 - b1;
            enforce(abs((b0 + b1) - 1) < 1e-5);
            outImage[x, y][] = b0 * a[x, y] + b1 * b[x, y];
        }
    }
    return outImage;
}

struct LaplacianPyramid {
    MirImage[] levels; // Index 0 is the lowest res as a delta from pitch black, the rest are deltas from the previous
}

LaplacianPyramid makePyramid(MirImage image, int levels) {
    auto minSize = 4 << levels;
    enforce(image.dims.width >= minSize && image.dims.height >= minSize);
    LaplacianPyramid pyramid;
    pyramid.levels ~= image;
    foreach (i; 1 .. levels) {
        auto last = pyramid.levels[$ - 1];
        auto half = last.downscale2;
        auto rec = half.upscale2(last.dims);
        pyramid.levels[$ - 1] = last.subtract(rec);
        pyramid.levels ~= half;
    }
    pyramid.levels.reverse();
    return pyramid;
}

MirImage reconstruct(LaplacianPyramid pyramid) {
    MirImage outImage = pyramid.levels[0];
    foreach (i; 1 .. pyramid.levels.length) {
        auto next = outImage.upscale2(pyramid.levels[i].dims);
        outImage = next.add(pyramid.levels[i]);
    }
    return outImage;
}

LaplacianPyramid blend(LaplacianPyramid pyr1, LaplacianPyramid pyr2) {
    enforce(pyr1.levels.length == pyr2.levels.length);
    enforce(pyr1.levels[$ - 1].dims == pyr2.levels[$ - 1].dims);
    auto blendFunc = (float x, float y) {
        return clamp(5 * x - 2f, 0f, 1f);
        // return x > 0.5 ? 1f : 0f;
        // size_t xGrid = cast(size_t) (x * 12);
        // size_t yGrid = cast(size_t) (y * 12);
        // return cast(float) ((xGrid ^ yGrid) & 1);
    };
    auto outPyr = LaplacianPyramid();
    foreach (i; 0 .. pyr1.levels.length) {
        outPyr.levels ~= performBlend(pyr1.levels[i], pyr2.levels[i], blendFunc.toDelegate);
    }
    return outPyr;
}

Image blend(Image im1, Image im2, int nLevels = 6) {
    enforce(im1.width == im2.width && im1.height == im2.height);
    auto pyr1 = im1.toMirImage.makePyramid(nLevels);
    auto pyr2 = im2.toMirImage.makePyramid(nLevels);
    auto blended = blend(pyr1, pyr2);
    return blended.reconstruct.toImage;
}

void main(string[] args) {
    enforce(args.length == 3);
    auto im1 = args[1].loadImageRgb;
    auto im2 = args[2].loadImageRgb;
    blend(im1, im2, 1).writeImageRgb("out.bmp");
}
