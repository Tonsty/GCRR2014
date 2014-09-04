#ifndef GCRR2014_H
#define GCRR2014_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <vector>
#include "lbfgs.h"

typedef pcl::PointXYZ PointType;
//typedef pcl::PointXYZRGB PointType;

typedef pcl::PointCloud<PointType> PointCloud;
typedef std::vector< PointCloud > PointClouds;
typedef PointCloud::Ptr PointCloudPtr;
typedef std::vector< PointCloudPtr > PointCloudPtrs;

struct Correspondence
{
	PointType targetPoint;
	PointType sourcePoint;
};
typedef std::vector<Correspondence> Correspondences;

struct Transformation 
{
	float roll;
	float pitch;
	float yaw;
	float x;
	float y;
	float z;
};
typedef std::vector<Transformation> Transformations;

class GCRR2014
{
public:
	GCRR2014();
	~GCRR2014();

	void setInputPointClouds(PointCloudPtrs &inputClouds);
	void setOverlapRelations(std::vector< std::vector<int> > overlapRelations);

	//total energy of overlapping areas
	float energy(Transformations transformations);
	//point-to-point energy
	float pair_energy(PointCloudPtr &target, PointCloudPtr &source,  
					  Transformation &targetTransform, Transformation &sourceTransform);

	PointCloudPtrs inputClouds;
	std::vector< std::vector<int> > overlapRelations;
};


#endif