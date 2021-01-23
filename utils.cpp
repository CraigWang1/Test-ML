/* 
 * All helper functions that are used to assist detection
 */

#include "utils.hpp"

void underwaterEnhance(cv::Mat &mat)
{
	double discard_ratio = 0.05;
	int hists[3][256];
	memset(hists, 0, 3 * 256 * sizeof(int));

	for (int y = 0; y < mat.rows; ++y)
	{
		uchar *ptr = mat.ptr<uchar>(y);
		for (int x = 0; x < mat.cols; ++x)
		{
			for (int j = 0; j < 3; ++j)
			{
				hists[j][ptr[x * 3 + j]] += 1;															            
			}
													        
		}
									    
	}

	int total = mat.cols * mat.rows;
	int vmin[3], vmax[3];
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 255; ++j)
		{
			hists[i][j + 1] += hists[i][j];
														        
		}
        vmin[i] = 0;
		vmax[i] = 255;
		while (hists[i][vmin[i]] < discard_ratio * total)
			vmin[i] += 1;
		while (hists[i][vmax[i]] > (1 - discard_ratio) * total)
			vmax[i] -= 1;
		if (vmax[i] < 255 - 1)
			vmax[i] += 1;										
	}
	for (int y = 0; y < mat.rows; ++y)
	{
		uchar *ptr = mat.ptr<uchar>(y);
		for (int x = 0; x < mat.cols; ++x)
		{
			for (int j = 0; j < 3; ++j)
			{
				int val = ptr[x * 3 + j];
				if (val < vmin[j])
					val = vmin[j];
				if (val > vmax[j])
					val = vmax[j];
				ptr[x * 3 + j] = static_cast<uchar>((val - vmin[j]) * 255.0 / (vmax[j] - vmin[j]));														
			}
		}
	}
}

float resize(cv::Mat &image, int image_size)
{
	/* Resize image while preserving aspect ratio. */
	int image_height = image.rows;        //initialize image height & width for convenience
	int image_width = image.cols;         
	int resized_height, resized_width;    //new img dims to downsize to
	float scale;                          //scale factor for resizing to original image

	// calculates what height and width to resize to preserve height/width ratios
	if (image_height > image_width) 
	{                 
		scale = (float)image_size / image_height;     //the taller side will become the image_size (eg. 512), so the scale is 512/og_h
		resized_height = image_size;                  //downsize taller side to 512
		resized_width = (int)(image_width * scale);   //scale the width down by same scale as the height, preserving side ratios
	}
	else 
	{                                           
		scale = (float)image_size / image_width;       //repeat same procedures as above except for width
		resized_height = (int)(image_height * scale);
		resized_width = image_size;
	}
	cv::resize(image, image, cv::Size(resized_width, resized_height));         //resize the image, keeping ratios
	return scale;
}

void preprocess(cv::Mat &image, std::vector<float> &img_data, int image_size)
{
	/*
	 * This is a helper function that returns an image_size x image_size image
	 * that is normalized for model input (necessary).
	 */
	// Assumes image is already downsized and enhanced from image acquisition; if it's not, process here
	if (image.cols != image_size)
	{
		resize(image, image_size);                      //model can only run specific size images
		underwaterEnhance(image);                       //avoid time complexity with small img
	}

	// create image_size x image_size input img
	cv::Mat temp;                                                              // use temp because we still need to log original img intact                                     
	cv::cvtColor(image, temp, cv::COLOR_BGR2RGB);                                     // copy and convert to rgb (model is trained on rgb)
	temp.convertTo(temp, CV_32FC3);                                            // converts to float matrix so we can divide later
	cv::Mat inp(image_size, image_size, CV_32FC3, cv::Scalar(128, 128, 128));  // makes input mat with shape (image_size, image_size, 3) filled with 128s
	temp.copyTo(inp(cv::Rect(0, 0, temp.cols, temp.rows)));                    // pastes the image on top left corner (point 0, 0) of empty cv mat

	// normalize image data
	cv::divide(inp, cv::Scalar(255.0, 255.0, 255.0), inp);          //convert to values from 0-1
	inp -= cv::Scalar(0.485, 0.456, 0.406);                         //subtract the mean from each channel
	cv::divide(inp, cv::Scalar(0.229, 0.224, 0.225), inp);          //divide each channel by standard deviation

	// copy the mat inside the std vector (model wrapper takes vector as input)
	img_data.assign((float*)inp.data, (float*)inp.data + inp.total()*inp.channels());       
}

void drawBox(cv::Mat &img, float xmin, float ymin, float xmax, float ymax, std::string classes[], int label, float score, std::vector<cv::Scalar> colors)
{
	/*
	* Helper function to aesthetically draw detected bounding box.
	*/
	std::string text = classes[label] + '-' + std::to_string(score);      //setup our label: eg. "bin-0.9996"
	cv::Scalar color = colors[label];                                     //assign color to class
	int baseline = 0;                                                     //baseline variable that the getTextSize function outputs
	cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);    //get our text size so we can be use it to draw aesthetic text
	cv::rectangle(img, {(int)xmin, (int)ymin}, {(int)xmax, (int)ymax}, color, 3);              //draws bbox
	cv::rectangle(img, {(int)xmin, (int)ymax - textSize.height - baseline},                    //draws a highlight behind text for ease of sight
	            {(int)xmin + textSize.width, (int)ymax}, color, -1);
	cv::putText(img, text, {(int)xmin, (int)ymax - baseline}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1);    //puts text on top of highlight
}