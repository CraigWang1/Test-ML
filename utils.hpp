/*
 * Hpp file for all of the helper functions that are used 
 */
#ifndef UTILS_HPP
#define UTILS_HPP

#include "opencv2/opencv.hpp"   
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <math.h>
#include <stdio.h>
#include <iostream>
#include <vector>    
#include <chrono>   

float resize(cv::Mat &, int);
void underwaterEnhance(cv::Mat &);
void preprocess(cv::Mat &, std::vector<float> &, int);
void drawBox(cv::Mat &, float, float, float, float, std::string[], int, float, std::vector<cv::Scalar>);

#endif