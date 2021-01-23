/*
 * Run model on video to test fps 
 */
#include "cppflow/include/Model.h"  
#include "cppflow/include/Tensor.h"
#include "utils.hpp"   

int main() 
{
    // load model
    Model model("model.pb");

    //IMPORTANT: Edit architecture of model (eg. efficientdet-d3 -> phi=3)
    int phi = 0; 
    int image_sizes[] = {512, 640, 768, 896, 1024, 1152, 1280};      //image sizes that effdet uses
    int image_size = image_sizes[phi];                                //takes the image size that our model uses
    std::string classes[] = {"gate"};                                //list of classes
    std::vector<cv::Scalar> colors = {{0, 255, 255}};                                 //setup our color (blue,green,red)

    // inititialize the model's input and output tensors
    auto inpName = new Tensor(model, "input_1");
    auto out_boxes = new Tensor(model, "filtered_detections/map/TensorArrayStack/TensorArrayGatherV3");
    auto out_scores = new Tensor(model, "filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3");
    auto out_labels = new Tensor(model, "filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3"); 

    cv::VideoCapture cap("video.mp4");

    // Default resolution of the frame is obtained.The default resolution is system dependent. 
    // int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH); 
    // int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT); 
    int fps = 25;

    // initialize video writer (for output)
    cv::VideoWriter out("result.mp4", cv::VideoWriter::fourcc('H','2','6','4'),
                           fps, cv::Size(512, 439));  
    /*
    cv::VideoWriter out("/content/result.mp4", cv::VideoWriter::fourcc('H','2','6','4'),
                           fps, cv::Size(frame_width, frame_height));  
    */

    int counter = 1;  //count which frame you're on
    // loop through the frames
    for (;;) 
    {
        auto start = std::chrono::high_resolution_clock::now();
      
        cv::Mat img;
        cv::Mat inp;
        std::vector<float> img_data;

        // process input image  
        cap >> img;
        if (img.empty()) 
        {
            std::cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        //std::cout << counter << std::endl;           //display frame number
        resize(img, image_size);              //resize img for faster enhance and to input model
        // underwaterEnhance(img);                      //underwater phoebe enhance for visualization
        cv::cvtColor(img, inp, cv::COLOR_BGR2RGB);          //convert from bgr img to rgb (model trained on rgb images) and copy to input img
        preprocess(inp, img_data, image_size);  //process image for input

        // Put data in tensor.
        inpName->set_data(img_data, {1,image_size,image_size,3});

        // run model
        model.run(inpName, { out_boxes, out_scores, out_labels });
        
        // convert tensors to std vectors
        auto boxes = out_boxes->get_data<float>();
        auto scores = out_scores->get_data<float>();
        auto labels = out_labels->get_data<int>();

        /*   Commented out because we're just writing 512 output, no need for og big output
        // scale output boxes back to original image
        for (int i=0; i<boxes.size(); i++) 
        {
            boxes[i] = boxes[i] / scale;
        }
        */

        // iterate over results and draw the boxes for visualization!
        for (int i=0; i<scores.size(); i++) 
        {
            if (scores[i] > 0.5) 
            {
                // extract output values
                float score = scores[i];
                int label = labels[i];       
                float xmin = boxes[i*4];    //the boxes come in 4 values: [xmin (left), ymin (top), xmax (right), ymax (bottom)]
                float ymin = boxes[i*4+1];
                float xmax = boxes[i*4+2];
                float ymax = boxes[i*4+3];

                // aesthetically visualize output box, label, and score
                drawBox(img, xmin, ymin, xmax, ymax, classes, label, score, colors);
            }
            else 
            {            //since the outputs are already sorted by score
                break;        //if it's lower than the thres then we know it's the last highest one,
            }                 //so we can afford to break out because all the other outputs will be below the threshold
        }
        out.write(img);
        // calculate fps
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
        int microseconds = duration.count();
        float seconds = (float)microseconds / 1000000;  // convert microseconds to seconds
        std::cout << "Frames per second: " << (1/seconds) << "\n";

        counter ++;
    }
    cap.release();
    out.release();

    return 0;
}