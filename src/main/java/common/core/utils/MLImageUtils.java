package common.core.utils;

import java.awt.FlowLayout;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.Random;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class MLImageUtils {

    public static abstract class MLImageTransform {

        public static class MLImageTransformCrop extends MLImageTransform {

            private int cropHeight;
            private int cropWidth;
            private Random random;

            public MLImageTransformCrop(final int heightP, final int widthP,
                                        final Random randomP) {
                this.cropHeight = heightP;
                this.cropWidth = widthP;
                this.random = randomP;

            }

            @Override
            public Mat apply(final Mat image) {
                int height = image.rows();
                int width = image.cols();

                if (this.random == null) {
                    this.random = new Random();
                }

                int cropX = 0;
                if (this.cropWidth >= width) {
                    // pad width
                    cropX = width - this.cropWidth;

                } else {
                    // randomly select x coordinate
                    cropX = random.nextInt(width - this.cropWidth);
                }

                int cropY = 0;
                if (this.cropHeight >= height) {
                    // pad height
                    cropY = height - this.cropHeight;

                } else {
                    // randomly select x coordinate
                    cropY = random.nextInt(height - this.cropHeight);
                }

                // pad image
                Mat imageTransformed = new Mat();
                if (cropX < 0 || cropY < 0) {
                    int left = Math.max(0, -cropX);
                    int top = Math.max(0, -cropY);

                    Core.copyMakeBorder(image, imageTransformed, top, 0, left,
                            0, Core.BORDER_CONSTANT, new Scalar(0));
                } else {
                    image.copyTo(imageTransformed);
                }

                // crop image
                if (cropX >= 0 || cropY >= 0) {
                    int x = Math.max(0, cropX);
                    int y = Math.max(0, cropY);

                    Rect roi = new Rect(x, y, this.cropWidth, this.cropHeight);
                    imageTransformed = new Mat(imageTransformed, roi);
                }

                image.release();
                return imageTransformed;
            }
        }

        public static class MLImageTransformFlip extends MLImageTransform {

            // Vertical flipping of the image (flipCode == 0) to switch between
            // top-left and bottom-left image origin. This is a typical
            // operation in video processing on Microsoft Windows* OS.
            // Horizontal flipping of the image with the subsequent horizontal
            // shift and absolute difference calculation to check for a
            // vertical-axis symmetry (flipCode > 0).
            // Simultaneous horizontal and vertical flipping of the image with
            // the subsequent shift and absolute difference calculation to check
            // for a central symmetry (flipCode < 0).
            // Reversing the order of point arrays (flipCode > 0 or flipCode ==
            // 0).
            private int flipCode;

            public MLImageTransformFlip(final int flipCodeP) {
                this.flipCode = flipCodeP;
            }

            @Override
            public Mat apply(final Mat image) {

                Mat imageTransformed = new Mat();
                Core.flip(image, imageTransformed, this.flipCode);

                image.release();
                return imageTransformed;
            }
        }

        public static class MLImageTransformJitter extends MLImageTransform {

            private float contrast;
            private float brightness;

            public MLImageTransformJitter(final float contrastP,
                                          final float brightnessP) {
                this.contrast = contrastP;
                this.brightness = brightnessP;
            }

            @Override
            public Mat apply(final Mat image) {
                Mat imageTransformed = new Mat();
                // image.copyTo(imageTransformed);
                image.convertTo(imageTransformed, -1, this.contrast,
                        this.brightness);
                image.release();
                return imageTransformed;
            }
        }

        public static class MLImageTransformResize extends MLImageTransform {

            private int newHeight;
            private int newWidth;
            private int interpolation;

            public MLImageTransformResize(final int newHeightP,
                                          final int newWidthP,
                                          final int interpolationP) {
                this.newHeight = newHeightP;
                this.newWidth = newWidthP;
                this.interpolation = interpolationP;
            }

            @Override
            public Mat apply(final Mat image) {
                Mat imageResized = new Mat();
                Imgproc.resize(image, imageResized,
                        new Size(this.newWidth, this.newHeight), 0, 0,
                        this.interpolation);

                image.release();
                return imageResized;
            }

        }

        public static class MLImageTransformRotate extends MLImageTransform {
            private int leftAngleBound;
            private int rightAngleBound;
            private Random random;

            public MLImageTransformRotate(final int leftAngleBoundP,
                                          final int rightAngleBoundP,
                                          final Random randomP) {
                this.leftAngleBound = leftAngleBoundP;
                this.rightAngleBound = rightAngleBoundP;
                this.random = randomP;

            }

            @Override
            public Mat apply(final Mat image) {
                int angle =
                        this.random.nextInt((this.rightAngleBound - this.leftAngleBound) + 1) + this.leftAngleBound;

                Mat rotationMat =
                        Imgproc.getRotationMatrix2D(new Point(image.cols() / 2,
                                image.rows() / 2), angle, 1);
                Mat imageRotated = new Mat();
                Imgproc.warpAffine(image, imageRotated, rotationMat,
                        new Size(image.cols(), image.rows()));

                image.release();
                return imageRotated;
            }
        }

        public abstract Mat apply(final Mat image);

    }

    public static void display(final Mat image) {
        JFrame frame = new JFrame();
        frame.getContentPane().setLayout(new FlowLayout());
        frame.getContentPane()
                .add(new JLabel(new ImageIcon(toBufferedImage(image))));
        frame.pack();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }

    public static Mat fromFlatArrayGreyscale(final float[] flat,
                                             final int offset,
                                             final int height,
                                             final int width) {
        Mat image = new Mat(height, width, CvType.CV_8UC1);

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                image.put(i, j, (int) flat[offset + i * width + j]);
            }
        }

        return image;
    }

    public static Mat fromFlatArrayRGB(final float[] flat, final int nRows,
                                       final int nCols) {

        Mat image = new Mat(nRows, nCols, CvType.CV_8UC3);

        int offsetR = 0;
        int offsetG = nRows * nCols;
        int offsetB = 2 * nRows * nCols;

        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                double[] pixel = new double[]{flat[offsetR + i * nCols + j],
                        flat[offsetG + i * nCols + j],
                        flat[offsetB + i * nCols + j]};
                image.put(i, j, pixel);
            }
        }

        return image;
    }

    public static Image toBufferedImage(final Mat m) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (m.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = m.channels() * m.cols() * m.rows();
        byte[] b = new byte[bufferSize];
        m.get(0, 0, b); // get all the pixels
        BufferedImage image = new BufferedImage(m.cols(), m.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster()
                .getDataBuffer()).getData();
        System.arraycopy(b, 0, targetPixels, 0, b.length);
        return image;

    }

    public static float[] toFlatArrayGreyscale(final Mat image) {
        int nRows = image.rows();
        int nCols = image.cols();
        float[] flat = new float[nRows * nCols];

        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                flat[i * nCols + j] = (float) image.get(i, j)[0];
            }
        }

        return flat;
    }

    public static float[] toFlatArrayRGB(final Mat image) {

        int nRows = image.rows();
        int nCols = image.cols();
        float[] flat = new float[image.channels() * nRows * nCols];

        int offsetR = 0;
        int offsetG = nRows * nCols;
        int offsetB = 2 * nRows * nCols;

        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                // double[] pixel2 = image.get(i, j);
                byte[] pixel = new byte[3];
                image.get(i, j, pixel);
                flat[offsetR + i * nCols + j] = (float) (pixel[0] & 0xFF);
                flat[offsetG + i * nCols + j] = (float) (pixel[1] & 0xFF);
                flat[offsetB + i * nCols + j] = (float) (pixel[2] & 0xFF);
            }
        }

        return flat;
    }

}
