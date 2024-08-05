#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdlib.h>
#include <string>
#include "Supp.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    string windowName;
    Mat threeImages[3], blueMask, canvasColor, canvasGray, srcI, hsv;
    char str[256];
    Point2i center;
    vector<Scalar> colors;
    int const MAXfPt = 200;
    int t1, t2, t3, t4;
    RNG rng(0);
    String imgPattern("Inputs/Traffic signs/Blue signs/*.png");
    vector<string> imageNames;

    // lower blue
    Scalar blueLower(100, 150, 0);
    Scalar blueUpper(140, 255, 255);

    // get MAXfPt random but brighter colors for drawing later
    for (int i = 0; i < MAXfPt; i++) {
        for (;;) {
            t1 = rng.uniform(0, 255); // blue
            t2 = rng.uniform(0, 255); // green
            t3 = rng.uniform(0, 255); // red
            t4 = t1 + t2 + t3;
            // Below get random colors that is not dim
            if (t4 > 255) break;
        }
        colors.push_back(Scalar(t1, t2, t3));
    }

    // Collect all image names satisfying the image name pattern
    cv::glob(imgPattern, imageNames, true);
    for (size_t i = 0; i < imageNames.size(); ++i) {
        srcI = imread(imageNames[i]);
        if (srcI.empty()) { // found no such file?
            cout << "cannot open image for reading" << endl;
            return -1;
        }

        // Open 2 large windows to display the results. One gives the detail. Other gives only the results
        int const noOfImagePerCol = 2, noOfImagePerRow = 3;
        Mat detailResultWin, win[noOfImagePerRow * noOfImagePerCol], legend[noOfImagePerRow * noOfImagePerCol];
        createWindowPartition(srcI, detailResultWin, win, legend, noOfImagePerCol, noOfImagePerRow);

        putText(legend[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
        putText(legend[1], "blueMask", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
        putText(legend[2], "Contours", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
        putText(legend[3], "Longest contour", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
        putText(legend[4], "Mask", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
        putText(legend[5], "Sign segmented", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

        int const noOfImagePerCol2 = 1, noOfImagePerRow2 = 2;
        Mat resultWin, win2[noOfImagePerRow2 * noOfImagePerCol2], legend2[noOfImagePerRow2 * noOfImagePerCol2];
        createWindowPartition(srcI, resultWin, win2, legend2, noOfImagePerCol2, noOfImagePerRow2);

        putText(legend2[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
        putText(legend2[1], "Sign segmented", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

        srcI.copyTo(win[0]);
        srcI.copyTo(win2[0]);

        // create canvases for drawing
        // canvasColor: for drawing each contour with different colors in win[2]
        // canvasGray: is used to do logical operation & being displayed for each
        // detected contour if the region in contour is large enough.
        canvasColor.create(srcI.rows, srcI.cols, CV_8UC3);
        canvasGray.create(srcI.rows, srcI.cols, CV_8U);
        canvasColor = Scalar(0, 0, 0);

        cvtColor(srcI, hsv, COLOR_BGR2HSV);
        inRange(hsv, blueLower, blueUpper, blueMask);
        cvtColor(blueMask, win[1], COLOR_GRAY2BGR); // show result of blue color

        // get contours of the blue regions
        vector<vector<Point>> contours;  // See tutorial note of its data structure
        findContours(blueMask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

        int index = -1, max = 0; // used to record the longest contour

        for (int j = 0; j < contours.size(); j++) { // We could have more than one sign in image
            canvasGray = 0;
            if (max < contours[j].size()) { // Find the longest contour as sign boundary
                max = contours[j].size();
                index = j;
            }
            drawContours(canvasColor, contours, j, colors[j % MAXfPt]); // draw boundaries
            drawContours(canvasGray, contours, j, 255);

            // The code below computes the center of the region
            Moments M = moments(canvasGray);
            center.x = M.m10 / M.m00;
            center.y = M.m01 / M.m00;

            // If the found center is not inside the contour, the result will be incorrect
            // Hence, improvement can be made here
            floodFill(canvasGray, center, 255); // fill inside sign boundary
            if (countNonZero(canvasGray) > 20) { // Check if the sign is too small
                sprintf_s(str, "Mask %d (area > 20)", j);
                imshow(str, canvasGray); // show only big enough sign
            }
        }
        canvasColor.copyTo(win[2]);
        if (index < 0) {
            waitKey();
            continue;
        }

        // Assume the longest contour (indicated by index) is the correct one
        // One should fine-tune this in real work
        canvasGray = 0;
        drawContours(canvasGray, contours, index, 255);
        cvtColor(canvasGray, win[3], COLOR_GRAY2BGR);

        Moments M = moments(canvasGray);
        center.x = M.m10 / M.m00;
        center.y = M.m01 / M.m00;

        // generate mask of the sign
        floodFill(canvasGray, center, 255); // fill inside sign boundary
        cvtColor(canvasGray, canvasGray, COLOR_GRAY2BGR);
        canvasGray.copyTo(win[4]);

        // use the mask to segment the color portion from image
        canvasColor = canvasGray & srcI;
        canvasColor.copyTo(win[5]);
        canvasColor.copyTo(win2[1]);

        windowName = "Segmentation of " + imageNames[i] + " (detail)";
        imshow(windowName, detailResultWin);
        imshow("Traffic sign segmentation", resultWin);

        waitKey();
        destroyAllWindows();
    }
    return 0;
}
