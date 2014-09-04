#include "gcrr2014.h"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>
#include <math.h>
#include <Eigen/Dense>

double overlapArea;
GCRR2014::GCRR2014(){}
GCRR2014::~GCRR2014(){}

void GCRR2014::setInputPointClouds(PointCloudPtrs &inputClouds)
{
	this->inputClouds = inputClouds; 
}

void GCRR2014::setOverlapRelations(std::vector< std::vector<int> > overlapRelations)
{
	this->overlapRelations = overlapRelations;
}

float GCRR2014::pair_energy(PointCloudPtr &target, PointCloudPtr &source, Transformation &targetTransform, Transformation &sourceTransform)
{
	//transform
	Eigen::Affine3f t_target,t_source;
	pcl::getTransformation(targetTransform.x,targetTransform.y,targetTransform.z,targetTransform.roll,targetTransform.pitch,targetTransform.yaw,t_target);
	pcl::getTransformation(sourceTransform.x,sourceTransform.y,sourceTransform.z,sourceTransform.roll,sourceTransform.pitch,sourceTransform.yaw,t_source);
	PointCloudPtr aftertranTarget ( new PointCloud );
	PointCloudPtr aftertranSource ( new PointCloud );
	pcl::transformPointCloud(*target, *aftertranTarget, t_target);
	pcl::transformPointCloud(*source, *aftertranSource, t_source);

	Correspondences correspondences;

	//kd-tree find closest point 
	 pcl::KdTreeFLANN<PointType> kdtree;
	 kdtree.setInputCloud(aftertranTarget);
	// std::cout<<"Kd-start!"<<std::endl;

	 for (int i=0; i<aftertranSource->points.size(); i++)
	 {
		 PointType searchPoint = aftertranSource->points[i];
		 if (  (searchPoint.x!=searchPoint.x)
			 ||(searchPoint.y!=searchPoint.y)
			 ||(searchPoint.z!=searchPoint.z))
		 {
			 continue;
		 }
		 std::vector<int> pointIdxNKNSearch;
		 std::vector<float> pointNKNSquaredDistance;

		 //std::cout<<"Kd-find nearset!"<<std::endl;

		 if ( ( kdtree.nearestKSearch (searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 ) )
		 {
			 Correspondence correspondence;
			 correspondence.sourcePoint = searchPoint;
			 correspondence.targetPoint = aftertranTarget->points[ pointIdxNKNSearch[0] ];
			 correspondences.push_back(correspondence);

			 // std::cout<<pointNKNSquaredDistance[0]<<std::endl;
		 }
	 }

	 //filter correspondences and optimize target point position
	// float threshold = 0.00001;
	 float threshold = 0.0001;
	 float temp_energy = 0;
	 Correspondences::iterator itr = correspondences.begin();
	 Correspondences correspondences2;
	 while(itr!=correspondences.end())
	 {
		 PointType targetPoint = itr->targetPoint;
		 PointType sourcePoint = itr->sourcePoint;
		 temp_energy = (targetPoint.x - sourcePoint.x) * (targetPoint.x - sourcePoint.x) +
			 (targetPoint.y - sourcePoint.y) * (targetPoint.y - sourcePoint.y) +
			 (targetPoint.z - sourcePoint.z) * (targetPoint.z - sourcePoint.z);
		 if (temp_energy < threshold)
		 {
			 correspondences2.push_back(*itr);
		 }
		 itr++;
	 }

	 //
	 float sub_energy = 0;
	 for ( int i = 0; i < correspondences2.size(); i++ )
	 {
		 PointType targetPoint = correspondences2[i].targetPoint;
		 PointType sourcePoint = correspondences2[i].sourcePoint;
		 sub_energy += (targetPoint.x - sourcePoint.x) * (targetPoint.x - sourcePoint.x) +
						(targetPoint.y - sourcePoint.y) * (targetPoint.y - sourcePoint.y) +
						 (targetPoint.z - sourcePoint.z) * (targetPoint.z - sourcePoint.z);
	 }

	overlapArea = correspondences2.size();
	if (overlapArea == 0)
		sub_energy += 9999;
	else
		sub_energy += 1/overlapArea * threshold;

	return sub_energy;
}