#include "stdafx.h"
#include "roi_align_layer.h"

#include <cfloat>
#include "caffe/fast_rcnn_layers.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

template <typename Dtype>
void ROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	ROIPoolingParameter roi_pool_param = this->layer_param_.roi_pooling_param();
	CHECK_GT(roi_pool_param.pooled_h(), 0)
		<< "pooled_h must be > 0";
	CHECK_GT(roi_pool_param.pooled_w(), 0)
		<< "pooled_w must be > 0";
	pooled_height_ = roi_pool_param.pooled_h();
	pooled_width_ = roi_pool_param.pooled_w();
	spatial_scale_ = roi_pool_param.spatial_scale();
	LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
	top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
		pooled_width_);
	max_idx_.Reshape(bottom[1]->num(), channels_, pooled_height_,
		pooled_width_);
}

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {

}