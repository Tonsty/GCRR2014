#include "gcrr2014.h"
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/cloud_viewer.h> 
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <stdlib.h>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <Eigen/Dense>
#include <pcl/PolygonMesh.h>
#include <pcl/io/vtk_lib_io.h>

#include <fstream>
ifstream fin("in.txt");
//#define cin fin

#define num 2
#define N 6
#define PI 3.1415926

GCRR2014 gcrr;
Transformations originalTransformations;
std::vector< std::vector< int > > overlapRelations(num);
std::vector< double > error;

lbfgsfloatval_t evaluateFunction(const lbfgsfloatval_t *x)
{
	lbfgsfloatval_t fx = 0.0;
	Transformations transformations;
	Transformation transformation;
	transformation.x = 0;
	transformation.y = 0;
	transformation.z = 0;
	transformation.roll = 0;
	transformation.pitch = 0;
	transformation.yaw = 0;
	transformations.push_back(transformation);

	for (int i=0 ; i<N ; i+=6)
	{
		transformation.x = x[i];
		transformation.y = x[i+1];
		transformation.z = x[i+2];
		transformation.roll = x[i+3];
		transformation.pitch = x[i+4];
		transformation.yaw = x[i+5];
		transformations.push_back(transformation);
	}

	for ( int i = 0; i < overlapRelations.size(); i++ )
	{
		int target_id = i;
		for ( int j = 0; j < overlapRelations[i].size(); j++ )
		{
			int source_id = overlapRelations[i][j];

			float part_energy = gcrr.pair_energy(gcrr.inputClouds[target_id], gcrr.inputClouds[source_id], 
				transformations[target_id], transformations[source_id]);
			fx += part_energy;
		}
	}

	return fx;
}

static lbfgsfloatval_t evaluateEnergy(	void *instance, 
										const lbfgsfloatval_t *x,
										lbfgsfloatval_t *g,
										const int n,
										const lbfgsfloatval_t step)
{
	lbfgsfloatval_t total_energy = 0.0;
	Transformations transformations;
	Transformation transformation;
	transformation.x = 0;
	transformation.y = 0;
	transformation.z = 0;
	transformation.roll = 0;
	transformation.pitch = 0;
	transformation.yaw = 0;
	transformations.push_back(transformation);

	for (int i=0 ; i<N ; i+=6)
	{
		transformation.x = x[i];
		transformation.y = x[i+1];
		transformation.z = x[i+2];
		transformation.roll = x[i+3];
		transformation.pitch = x[i+4];
		transformation.yaw = x[i+5];
		transformations.push_back(transformation);
	}

	for ( int i = 0; i < overlapRelations.size(); i++ )
	{
		int target_id = i;
		for ( int j = 0; j < overlapRelations[i].size(); j++ )
		{
			int source_id = overlapRelations[i][j];

			float part_energy = gcrr.pair_energy(gcrr.inputClouds[target_id], gcrr.inputClouds[source_id], 
				transformations[target_id], transformations[source_id]);

			std::cout<<"overlapRelations "<<target_id<<" "<<source_id<<std::endl;
			total_energy += part_energy;
		}
	}

	//calculate guide
	float delta1 = 0.00000001, delta2 =0.00001/180 * PI;
	lbfgsfloatval_t *x2 = lbfgs_malloc(N);
	for (int i=0 ; i<N ; i++)
	{
		x2[i] = x[i];
	}
	for (int i=0 ; i<N ; i++)
	{
		if (i%6 < 3)
		{
			x2[i] += delta1;
			g[i] = (evaluateFunction(x2)-evaluateFunction(x))/delta1;
			x2[i] -= delta1;
		}
		else
		{
			x2[i] += delta2;
			g[i] = (evaluateFunction(x2)-evaluateFunction(x))/delta2;
			x2[i] -= delta2;
		}
	}

	return total_energy;
}

static int progress(
	void *instance,
	const lbfgsfloatval_t *x,
	const lbfgsfloatval_t *g,
	const lbfgsfloatval_t fx,
	const lbfgsfloatval_t xnorm,
	const lbfgsfloatval_t gnorm,
	const lbfgsfloatval_t step,
	int n,
	int k,
	int ls
	)
{
	printf("Iteration %d:\n", k);
	error.push_back(fx);
	printf("  fx = %f, x[0] = %f, x[1] = %f, x[2] = %f, x[3] = %f, x[4] = %f, x[5] = %f\n", fx, x[0], x[1], x[2], x[3], x[4], x[5]);
	printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
	printf("\n");
	return 0;
}

Transformations compute()
{
	int ret = 0;
	lbfgsfloatval_t fx;
	lbfgsfloatval_t *x = lbfgs_malloc(N);
	lbfgs_parameter_t param;

	if (x == NULL) {
		printf("ERROR: Failed to allocate a memory block for variables.\n");
	}
	else
		printf("Success: allocate a memory block for variables.\n");

	/* Initialize the variables. */
	for (int i = 0;i < N;i += 6) {
		x[i] = originalTransformations[i/6 + 1].x;
		x[i+1] = originalTransformations[i/6 + 1].y;
		x[i+2] = originalTransformations[i/6 + 1].z;
		x[i+3] = originalTransformations[i/6 + 1].roll;
		x[i+4] = originalTransformations[i/6 + 1].pitch;
		x[i+5] = originalTransformations[i/6 + 1].yaw;
	}

	/* Initialize the parameters for the L-BFGS optimization. */
	lbfgs_parameter_init(&param);
	/*param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;*/

	/*
    Start the L-BFGS optimization; this will invoke the callback functions
    evaluate() and progress() when necessary.
     */
    ret = lbfgs(N, x, &fx, evaluateEnergy, progress, NULL, &param);
	printf("Has already L_BFGS initilized.\n");

	Transformations transformations;
	Transformation transformation;
	for (int i=0 ; i<N ; i+=6)
	{
		transformation.x = x[i];
		transformation.y = x[i+1];
		transformation.z = x[i+2];
		transformation.roll = x[i+3];
		transformation.pitch = x[i+4];
		transformation.yaw = x[i+5];
		transformations.push_back(transformation);
	}

	return transformations;
}

int main(int argc, char** argv)
{
	// //PointCloudPtr cloud (new PointCloud);
	// //pcl::io::loadPLYFile<PointType> ("dragon.ply", *cloud);

	// pcl::PolygonMesh mesh;
	// //pcl::io::loadPolygonFilePLY("bun045.ply",mesh);
	// pcl::io::loadPolygonFile("dragon.ply",mesh);
	// std::cout<<"OK"<<endl;
	// boost::shared_ptr<pcl::visualization::PCLVisualizer>viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	// viewer->setBackgroundColor(0,0,0);
	// //viewer->addModelFromPLYFile("dragon.ply");

	
	// viewer->addPolygonMesh(mesh,"0");
	
	// viewer->spin();
	// // while (!viewer->wasStopped ())  
 // // 	{    
 // // 		viewer->spinOnce (100);
 // // 		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
 // // 	} 
// 
	PointCloudPtrs inputClouds;
	//load files
	for (int i=0; i<num; i++)
	{
		PointCloudPtr cloud (new PointCloud);
		int respond = 0;
		switch (i)
		{
		case 0:	respond = pcl::io::loadPLYFile<PointType> ("bun045.ply", *cloud);
				std::cout<<"Succeed Load!--bun045.ply"<<endl;
			break;
		case 1:	respond = pcl::io::loadPLYFile<PointType> ("bun000.ply", *cloud);
				std::cout<<"Succeed Load!--bun000.ply"<<endl;
			break;
/*		case 2:	respond = pcl::io::loadPLYFile<PointType> ("bun090.ply", *cloud);
				//respond = pcl::io::loadPCDFile<PointType> ("pcd2.pcd", *cloud);
				std::cout<<"Succeed Load!--bun090.ply"<<endl;
			break;
		case 3:	respond = pcl::io::loadPLYFile<PointType> ("bun180.ply", *cloud);
				//respond = pcl::io::loadPCDFile<PointType> ("pcd3.pcd", *cloud);
				std::cout<<"Succeed Load!--bun180.ply"<<endl;
			break;
	
		case 4:	respond = pcl::io::loadPLYFile<PointType> ("bun270.ply", *cloud);
				//respond = pcl::io::loadPCDFile<PointType> ("pcd3.pcd", *cloud);
			std::cout<<"Succeed Load!--bun270.ply"<<endl;
			break;
		case 5:	respond = pcl::io::loadPLYFile<PointType> ("bun315.ply", *cloud);
				//respond = pcl::io::loadPCDFile<PointType> ("pcd3.pcd", *cloud);
			std::cout<<"Succeed Load!--bun315.ply"<<endl;
			break;
		case 6:	respond = pcl::io::loadPLYFile<PointType> ("chin.ply", *cloud);
				//respond = pcl::io::loadPCDFile<PointType> ("pcd3.pcd", *cloud);
			std::cout<<"Succeed Load!--chin.ply"<<endl;
			break;
		case 7:	respond = pcl::io::loadPLYFile<PointType> ("ear_back.ply", *cloud);
				//respond = pcl::io::loadPCDFile<PointType> ("pcd3.pcd", *cloud);
			std::cout<<"Succeed Load!--ear_back.ply"<<endl;
			break;
		case 8:	respond = pcl::io::loadPLYFile<PointType> ("top2.ply", *cloud);
				//respond = pcl::io::loadPCDFile<PointType> ("pcd3.pcd", *cloud);
			std::cout<<"Succeed Load!--top2.ply"<<endl;
			break;
		case 9:	respond = pcl::io::loadPLYFile<PointType> ("top3.ply", *cloud);
				//respond = pcl::io::loadPCDFile<PointType> ("pcd3.pcd", *cloud);
			std::cout<<"Succeed Load!--top3.ply"<<endl;
			break;
*/		
		}
		if (respond == -1)
		{
			PCL_ERROR ("Couldn't read file \n");
			system("pause");
			return(-1);
		}
		inputClouds.push_back(cloud);
	}
	
	//load transformations
	for (int i=0 ; i<inputClouds.size() ; i++)
	{
		Transformation transformation;
		//std::cin
		fin
			>>transformation.x>>transformation.y>>transformation.z
			>>transformation.roll>>transformation.pitch>>transformation.yaw;
		originalTransformations.push_back(transformation);
	}
	std::cout<<"Succeed Load!--OriginalTransformation"<<endl;
	
	//transformation
	Eigen::Affine3f t[num];
	PointCloudPtr aftertran[num];
	for (int i=0;i<inputClouds.size();i++)
	{
		pcl::getTransformation(originalTransformations[i].x,originalTransformations[i].y,originalTransformations[i].z,
			originalTransformations[i].roll,originalTransformations[i].pitch,originalTransformations[i].yaw,t[i]);
		aftertran[i]=PointCloudPtr(new PointCloud);
		pcl::transformPointCloud(*inputClouds[i],*aftertran[i], t[i]);
	}
	std::cout<<"Succeed Transform!"<<endl;

	//visualization
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Point Clouds"));
	viewer->initCameraParameters();
	int v1(0);
	viewer->createViewPort(0,0,0.5,1,v1);
	viewer->setBackgroundColor(0,0,0,v1);
	viewer->addText("Original",20,20,"v1",v1);
	viewer->addCoordinateSystem(0.01,v1);
	int v2(0);
	viewer->createViewPort(0.5,0,1,1,v2);
	viewer->setBackgroundColor(0.1,0.1,0.1,v2);
	viewer->addText("AfterTran",20,20,"v2",v2);
	viewer->addCoordinateSystem(0.01,v2);

//	pcl::visualization::PointCloudColorHandlerCustom<PointType> zero(inputClouds[0],0,255,0);
	//pcl::visualization::createPolygon<PointType> zero(inputClouds[0]);
	//viewer->addPointCloud(inputClouds[0],zero,"0",v1);
/*	pcl::visualization::PointCloudColorHandlerCustom<PointType> one(inputClouds[1],255,255,255);
	viewer->addPointCloud(inputClouds[1],one,"1",v1);
	pcl::visualization::PointCloudColorHandlerCustom<PointType> two(inputClouds[2],0,0,255);
	viewer->addPointCloud(inputClouds[2],two,"2",v1);
	pcl::visualization::PointCloudColorHandlerCustom<PointType> three(inputClouds[3],255,255,255);
	viewer->addPointCloud(inputClouds[3],three,"3",v1);
	
	pcl::visualization::PointCloudColorHandlerCustom<PointType> four(inputClouds[4],255,255,102);
	viewer->addPointCloud(inputClouds[4],four,"4",v1);
	pcl::visualization::PointCloudColorHandlerCustom<PointType> five(inputClouds[5],255,102,102);
	viewer->addPointCloud(inputClouds[5],five,"5",v1);
	pcl::visualization::PointCloudColorHandlerCustom<PointType> six(inputClouds[6],0,0,255);
	viewer->addPointCloud(inputClouds[6],six,"6",v1);
	pcl::visualization::PointCloudColorHandlerCustom<PointType> seven(inputClouds[7],204,204,255);
	viewer->addPointCloud(inputClouds[7],seven,"7",v1);
	pcl::visualization::PointCloudColorHandlerCustom<PointType> eight(inputClouds[8],255,255,102);
	viewer->addPointCloud(inputClouds[8],eight,"8",v1);
	pcl::visualization::PointCloudColorHandlerCustom<PointType> nine(inputClouds[9],204,51,153);
	viewer->addPointCloud(inputClouds[9],nine,"9",v1);
*/	
	pcl::visualization::PointCloudColorHandlerCustom<PointType> zerot(aftertran[0],0,255,0);
	viewer->addPointCloud(aftertran[0],zerot,"10",v2);
/*	pcl::visualization::PointCloudColorHandlerCustom<PointType> onet(aftertran[1],255,255,255);
	viewer->addPointCloud(aftertran[1],onet,"11",v2);
	pcl::visualization::PointCloudColorHandlerCustom<PointType> twot(aftertran[2],0,0,255);
	viewer->addPointCloud(aftertran[2],twot,"12",v2);
	pcl::visualization::PointCloudColorHandlerCustom<PointType> threet(aftertran[3],255,255,255);
	viewer->addPointCloud(aftertran[3],threet,"13",v2);
	
	pcl::visualization::PointCloudColorHandlerCustom<PointType> fourt(aftertran[4],255,255,102);
	viewer->addPointCloud(aftertran[4],fourt,"14",v2);
	pcl::visualization::PointCloudColorHandlerCustom<PointType> fivet(aftertran[5],255,102,102);
	viewer->addPointCloud(aftertran[5],fivet,"15",v2);
	pcl::visualization::PointCloudColorHandlerCustom<PointType> sixt(aftertran[6],0,0,255);
	viewer->addPointCloud(aftertran[6],sixt,"16",v2);
	pcl::visualization::PointCloudColorHandlerCustom<PointType> sevent(aftertran[7],204,204,255);
	viewer->addPointCloud(aftertran[7],sevent,"17",v2);
	pcl::visualization::PointCloudColorHandlerCustom<PointType> eightt(aftertran[8],255,255,102);
	viewer->addPointCloud(aftertran[8],eightt,"18",v2);
	pcl::visualization::PointCloudColorHandlerCustom<PointType> ninet(aftertran[9],204,51,153);
	viewer->addPointCloud(aftertran[9],ninet,"19",v2);
*/
	while (!viewer->wasStopped ())  
	{    
		viewer->spinOnce (100);
		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	}  
	std::cout<<"Before close!"<<endl;
	viewer->removeAllPointClouds();
	viewer->resetCamera();
	viewer->~PCLVisualizer();
	std::cout<<"After close!"<<endl;
	
	//like a chain list to save overlap relations
	int c;
	for (int i=0;i<num;i++)
	{
		for (int j=0;j<num;j++)
		{
			fin>>c;
			if (c==1)
			{
				overlapRelations[i].push_back(j);
			}
		}
	}


	gcrr.setInputPointClouds( inputClouds );
	gcrr.setOverlapRelations(overlapRelations);


	Transformations finalTransformations = compute();
	for (int i=0;i<error.size();i++)
	{
		std::cout<<"Optimization "<<i+1<<" : error = "<<error[i]<<std::endl;
	}
	std::cout<<endl;
	std::cout<<"finalTransformations:"<<endl;
	std::cout<<"0  0  0  0  0  0"<<endl;
	//cout final transformation for point clouds
	for (int i=0; i<finalTransformations.size(); i++)
	{
		std::cout<<setprecision(9)<<setiosflags(ios::fixed)
				 <<finalTransformations[i].x<<"  "<<finalTransformations[i].y<<"  "<<finalTransformations[i].z<<"  "
				 <<finalTransformations[i].roll<<"  "<<finalTransformations[i].pitch<<"  "<<finalTransformations[i].yaw
				 <<std::endl;
		pcl::getTransformation(finalTransformations[i].x,finalTransformations[i].y,finalTransformations[i].z,
			finalTransformations[i].roll,finalTransformations[i].pitch,finalTransformations[i].yaw,t[i+1]);
	}

	PointCloudPtr afterLBFGS[num];
	for (int i=0;i<inputClouds.size();i++)
	{
		afterLBFGS[i]=PointCloudPtr(new PointCloud);
		pcl::transformPointCloud(*inputClouds[i],*afterLBFGS[i], t[i]);
	}

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer2(new pcl::visualization::PCLVisualizer("LBFGS TRAN"));
	viewer2->initCameraParameters();

	int v3(0);
	viewer2->createViewPort(0,0,0.5,1,v3);
	viewer2->setBackgroundColor(0,0,0,v3);
	viewer2->addText("BeforeLBFGS",10,10,"v3",v3);
	viewer2->addCoordinateSystem(0.01,v3);
	int v4(0);
	viewer2->createViewPort(0.5,0,1,1,v4);
	viewer2->setBackgroundColor(0.1,0.1,0.1,v4);
	viewer2->addText("AfterLBFGS",10,10,"v4",v4);
	viewer2->addCoordinateSystem(0.01,v4);

	viewer2->addPointCloud(aftertran[0],zerot,"10",v3);
/*	viewer2->addPointCloud(aftertran[1],onet,"11",v3);
	viewer2->addPointCloud(aftertran[1],twot,"12",v3);
	viewer2->addPointCloud(aftertran[2],threet,"13",v3);

	viewer2->addPointCloud(aftertran[1],fourt,"14",v3);
	viewer2->addPointCloud(aftertran[2],fivet,"15",v3);
	viewer2->addPointCloud(aftertran[3],sixt,"16",v3);
	viewer2->addPointCloud(aftertran[7],sevent,"17",v3);
	viewer2->addPointCloud(aftertran[1],eightt,"18",v3);
	viewer2->addPointCloud(aftertran[2],ninet,"19",v3);
*/
	pcl::visualization::PointCloudColorHandlerCustom<PointType> zerott(afterLBFGS[0],0,255,0);
	viewer2->addPointCloud(afterLBFGS[0],zerott,"20",v4);
/*	pcl::visualization::PointCloudColorHandlerCustom<PointType> onett(afterLBFGS[1],255,255,255);
	viewer2->addPointCloud(afterLBFGS[1],onett,"21",v4);
	pcl::visualization::PointCloudColorHandlerCustom<PointType> twott(afterLBFGS[1],0,0,255);
	viewer2->addPointCloud(afterLBFGS[1],twott,"22",v4);
	pcl::visualization::PointCloudColorHandlerCustom<PointType> threett(afterLBFGS[2],255,255,255);
	viewer2->addPointCloud(afterLBFGS[2],threett,"23",v4);
	
	pcl::visualization::PointCloudColorHandlerCustom<PointType> fourtt(afterLBFGS[1],255,255,102);
	viewer2->addPointCloud(afterLBFGS[1],fourtt,"24",v4);
	pcl::visualization::PointCloudColorHandlerCustom<PointType> fivett(afterLBFGS[2],255,102,102);
	viewer2->addPointCloud(afterLBFGS[2],fivett,"25",v4);
	pcl::visualization::PointCloudColorHandlerCustom<PointType> sixtt(afterLBFGS[3],0,0,255);
	viewer2->addPointCloud(afterLBFGS[3],sixtt,"26",v4);
	pcl::visualization::PointCloudColorHandlerCustom<PointType> seventt(afterLBFGS[7],204,204,255);
	viewer2->addPointCloud(afterLBFGS[7],seventt,"27",v4);
	pcl::visualization::PointCloudColorHandlerCustom<PointType> eighttt(afterLBFGS[1],255,255,102);
	viewer2->addPointCloud(afterLBFGS[1],eighttt,"28",v4);
	pcl::visualization::PointCloudColorHandlerCustom<PointType> ninett(afterLBFGS[2],204,51,153);
	viewer2->addPointCloud(afterLBFGS[2],ninett,"29",v4);
*/	
	while (!viewer2->wasStopped ())  
	{    
		viewer2->spinOnce (100);
		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	}
	viewer2->removeAllPointClouds();
	viewer2->resetCamera();
	viewer2->~PCLVisualizer();
	

	system("pause");
	return 0;
}