// MobileNet_V1_IR_Sample.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#ifdef _DEBUG
#pragma comment(lib,"cpu_extension.lib")
#pragma comment(lib,"inference_engined.lib")
#pragma comment(lib,"tbb_debug.lib")
#pragma comment(lib,"opencv_core412d.lib")
#pragma comment(lib,"opencv_imgcodecs412d.lib")
#pragma comment(lib,"opencv_imgproc412d.lib")
#else
#pragma comment(lib,"cpu_extension.lib")
#pragma comment(lib,"inference_engine.lib")
#pragma comment(lib,"tbb.lib")
#pragma comment(lib,"opencv_core412.lib")
#pragma comment(lib,"opencv_imgcodecs412.lib")
#pragma comment(lib,"opencv_imgproc412.lib")
#endif

#include <inference_engine.hpp> 
#include <iostream>
#include <fstream>
#include <string> 
#include <vector>
#include <opencv2/opencv.hpp>

using namespace InferenceEngine;
using namespace std;

string inputName;
string outputName;
InferRequest inferReq;
vector<string> labels;

void initModel(string xml, string bin)
{
	try
	{
		Core ie;
		CNNNetReader network_reader;
		network_reader.ReadNetwork(xml);
		network_reader.ReadWeights(bin);
		network_reader.getNetwork().setBatchSize(1);
		CNNNetwork network = network_reader.getNetwork();
		inputName = network.getInputsInfo().begin()->first;
		InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;

		input_info->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
		input_info->setLayout(Layout::NCHW);
		input_info->setPrecision(Precision::U8);

		outputName = network.getOutputsInfo().begin()->first;
		DataPtr output_info = network.getOutputsInfo().begin()->second;
		output_info->setPrecision(Precision::FP16);

		ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU");
		inferReq = executable_network.CreateInferRequest();
	}
	catch (const std::exception & ex)
	{
		std::cerr << ex.what() << std::endl;
	}
}

void  matU8ToBlob(const cv::Mat& orig_image, InferenceEngine::Blob::Ptr& blob, int batchIndex = 0)
{
	InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();
	const size_t width = blobSize[3];
	const size_t height = blobSize[2];
	const size_t channels = blobSize[1];
	uint8_t* blob_data = blob->buffer().as<uint8_t*>();

	cv::Mat resized_image(orig_image);
	if (static_cast<int>(width) != orig_image.size().width ||
		static_cast<int>(height) != orig_image.size().height)
	{
		cv::resize(orig_image, resized_image, cv::Size(width, height));
	}

	int batchOffset = batchIndex * width * height * channels;

	for (size_t c = 0; c < channels; c++)
	{
		for (size_t h = 0; h < height; h++)
		{
			for (size_t w = 0; w < width; w++)
			{
				blob_data[batchOffset + c * width * height + h * width + w] =
					resized_image.at<cv::Vec3b>(h, w)[c];
			}
		}
	}
}

string infer(cv::Mat rgb, float& rtP)
{
	Blob::Ptr imgBlob = inferReq.GetBlob(inputName);
	matU8ToBlob(rgb, imgBlob);
	inferReq.Infer();
	Blob::Ptr output = inferReq.GetBlob(outputName);
	float* logits = output->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();

	int maxIdx = 0;
	float maxP = 0;
	int nclasses = labels.size();
	float sum = 1;

	for (int i = 0; i < nclasses; i++)
	{
		logits[i] = exp(logits[i]);
		sum = sum + logits[i];
		if (logits[i] > maxP)
		{
			maxP = logits[i];
			maxIdx = i;
		}
	}

	rtP = maxP / sum;
	return labels[maxIdx];
}

void readLabel(string labelPath)
{
	ifstream in(labelPath);
	string line;
	if (in)
	{
		while (getline(in, line))
		{
			labels.push_back(line);
		}
	}
}

int main(int argc, char * argv[])
{
	if (argc < 2)
	{
		cout << "Usage: this.exe cat.jpg" << endl;
	}
	string img = argc < 2 ? "E:\\Common\\cat.jpg" : argv[1];
	string xml = "E:\\Deposit\\TensorFlowModel\\mobilenet_v1_1.0_224\\ir\\FP16\\nosqueeze\\mobilenet_v1_1.0_224_frozen.xml";
	string bin = "E:\\Deposit\\TensorFlowModel\\mobilenet_v1_1.0_224\\ir\\FP16\\nosqueeze\\mobilenet_v1_1.0_224_frozen.bin";
	initModel(xml, bin);
	string label = "E:\\Deposit\\TensorFlowModel\\ImageNetLabels\\synset_words_chi.txt";
	readLabel(label);
	cv::Mat rgb = cv::imread(img);
	cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
	float p;
	string cls = infer(rgb, p);
	cout << cls << " " << p << endl;
}

