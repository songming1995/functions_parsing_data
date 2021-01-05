// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cmath>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <queue>
#include <utility>
#include <vector>
#include <numeric>
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "lidarFactor.hpp"
#include <iomanip>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

/*#include <mrpt/poses/CPoint3D.h>
#include <mrpt/poses/CPointPDFGaussian.h>
#include <mrpt/poses/CPose3D.h>
#include <mrpt/poses/CPoses2DSequence.h>
#include <mrpt/system/CTicTac.h> */
#include <iostream>
using namespace std;
#define DISTORTION 0


int corner_correspondence = 0, plane_correspondence = 0;
std::vector<ceres::ResidualBlockId> psrIDs;

constexpr double SCAN_PERIOD = 0.1;
constexpr double DISTANCE_SQ_THRESHOLD = 25;
constexpr double NEARBY_SCAN = 2.5;

int skipFrameNum = 5;
bool systemInited = false;
int framenum =0;

double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;

pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());

pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

int laserCloudCornerLastNum = 0;
int laserCloudSurfLastNum = 0;

// Transformation from current frame to world frame
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);

// q_curr_last(x, y, z, w), t_curr_last
double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};

Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLessSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLessFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;
std::mutex mBuf;

// undistort lidar point
void TransformToStart(PointType const *const pi, PointType *const po)
{
    //interpolation ratio
    double s;
    if (DISTORTION)
        s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
    else
        s = 1.0;
    //s = 1;
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
    Eigen::Vector3d t_point_last = s * t_last_curr;
    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;

    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}

// transform all lidar points to the start of the next frame

void TransformToEnd(PointType const *const pi, PointType *const po)
{
    // undistort point first
    pcl::PointXYZI un_point_tmp;
    TransformToStart(pi, &un_point_tmp);

    Eigen::Vector3d un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z);
    Eigen::Vector3d point_end = q_last_curr.inverse() * (un_point - t_last_curr);

    po->x = point_end.x();
    po->y = point_end.y();
    po->z = point_end.z();

    //Remove distortion time info
    po->intensity = int(pi->intensity);
}

void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2)
{
    mBuf.lock();
    cornerSharpBuf.push(cornerPointsSharp2);
    mBuf.unlock();
}

void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2)
{
    mBuf.lock();
    cornerLessSharpBuf.push(cornerPointsLessSharp2);
    mBuf.unlock();
}

void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2)
{
    mBuf.lock();
    surfFlatBuf.push(surfPointsFlat2);
    mBuf.unlock();
}

void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2)
{
    mBuf.lock();
    surfLessFlatBuf.push(surfPointsLessFlat2);
    mBuf.unlock();
}

//receive all point cloud
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
    mBuf.lock();
    fullPointsBuf.push(laserCloudFullRes2);
    mBuf.unlock();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserOdometry");
    ros::NodeHandle nh;

    nh.param<int>("mapping_skip_frame", skipFrameNum, 2);

    printf("Mapping %d Hz \n", 10 / skipFrameNum);

    ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100, laserCloudSharpHandler);

    ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100, laserCloudLessSharpHandler);

    ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, laserCloudFlatHandler);

    ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100, laserCloudLessFlatHandler);

    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100, laserCloudFullResHandler);

    ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);

    ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);

    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);

    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);

    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);

    nav_msgs::Path laserPath;

    int frameCount = 0;
    ros::Rate rate(100);

    while (ros::ok())
    {
        ros::spinOnce();

        if (!cornerSharpBuf.empty() && !cornerLessSharpBuf.empty() &&
            !surfFlatBuf.empty() && !surfLessFlatBuf.empty() &&
            !fullPointsBuf.empty())
        {
            timeCornerPointsSharp = cornerSharpBuf.front()->header.stamp.toSec();
            timeCornerPointsLessSharp = cornerLessSharpBuf.front()->header.stamp.toSec();
            timeSurfPointsFlat = surfFlatBuf.front()->header.stamp.toSec();
            timeSurfPointsLessFlat = surfLessFlatBuf.front()->header.stamp.toSec();
            timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp.toSec();

            if (timeCornerPointsSharp != timeLaserCloudFullRes ||
                timeCornerPointsLessSharp != timeLaserCloudFullRes ||
                timeSurfPointsFlat != timeLaserCloudFullRes ||
                timeSurfPointsLessFlat != timeLaserCloudFullRes)
            {
                printf("unsync messeage!");
                ROS_BREAK();
            }

            mBuf.lock();
            cornerPointsSharp->clear();
            pcl::fromROSMsg(*cornerSharpBuf.front(), *cornerPointsSharp);
            cornerSharpBuf.pop();

            cornerPointsLessSharp->clear();
            pcl::fromROSMsg(*cornerLessSharpBuf.front(), *cornerPointsLessSharp);
            cornerLessSharpBuf.pop();

            surfPointsFlat->clear();
            pcl::fromROSMsg(*surfFlatBuf.front(), *surfPointsFlat);
            surfFlatBuf.pop();

            surfPointsLessFlat->clear();
            pcl::fromROSMsg(*surfLessFlatBuf.front(), *surfPointsLessFlat);
            surfLessFlatBuf.pop();

            laserCloudFullRes->clear();
            pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullRes);
            fullPointsBuf.pop();
            mBuf.unlock();

            TicToc t_whole;
            // initializing
            if (!systemInited)
            {
                systemInited = true;
                std::cout << "Initialization finished \n";

                string filename = "/home/songming/lidarabsodometry.txt";
                std::cout << "The first frame is the reference frame, save 6D pose in " << filename << " ..." << std::endl;
                ofstream f;
                f.open(filename.c_str(),ios::app);
                f << fixed;
                f << setprecision(9) << 1 << " " << 0 << " " << 0 << " "  << 0 << " " <<
                                        0 << " " << 1 << " " << 0 << " "  << 0 << " " <<
                                        0 << " " << 0 << " " << 1 << " "  << 0 << std::endl; 
                f.close();

                string filename113 = "/home/songming/lidarrelodometry.txt";
                ofstream f54;
                f54.open(filename113.c_str(),ios::app);
                f54 << fixed;
                f54 << setprecision(9) << 0 << " " << 0  << " " << 0 << " " <<
                                        1 << " " << 0 << " " << 0 << " "  << 0 << std::endl;
                f54.close();

                string filename1 = "/home/songming/lodometry_cov.txt";
                ofstream f23;
                f23.open(filename1.c_str(),ios::app);
                f23 << fixed;
                f23 << setprecision(9) <<
                1e-19 << " " << 1e-19 << " " << 1e-19 << " "  << 1e-19 << " " << 1e-19 << " " << 1e-19 << " "  << 1e-19 << " "<<
                1e-19 << " " << 1e-19  << " " << 1e-19 << " "  << 1e-19 << " " << 1e-19  << " " << 1e-19 << " "<< 1e-19 << " "<<
                1e-19 << " " << 1e-19 << " " << 1e-19 << " "  << 1e-19 << " " << 1e-19  << " " << 1e-19 << " " << 1e-19 << " "<<
                1e-19 << " " << 1e-19  << " " << 1e-19 << " "  << 1e-19 << " " << 1e-19 << " " << 1e-19 << " " << 1e-19 << " "<<
                1e-19 << " " << 1e-19  << " " << 1e-19 << " "  << 1e-19 << " " << 1e-19  << " " << 1e-19 << " " << 1e-19 << " "<<
                1e-19 << " " << 1e-19  << " " <<1e-19 << " "  << 1e-19 << " " << 1e-19  << " " << 1e-19 << " " << 1e-19 << " "<<
                1e-19 << " " << 1e-19 << " " << 1e-19 << " "  << 1e-19 << " " << 1e-19  << " " << 1e-19 << " " << 1e-19 << std::endl;
                f23.close();
            }
            else
            {
                int cornerPointsSharpNum = cornerPointsSharp->points.size();
                int surfPointsFlatNum = surfPointsFlat->points.size();

                TicToc t_opt;
                for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter)
                {
                    corner_correspondence = 0;
                    plane_correspondence = 0;

                    //ceres::LossFunction *loss_function = NULL;
                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                    ceres::LocalParameterization *q_parameterization =
                        new ceres::EigenQuaternionParameterization();
                    ceres::Problem::Options problem_options;

                    ceres::Problem problem(problem_options);
                    problem.AddParameterBlock(para_q, 4, q_parameterization);
                    problem.AddParameterBlock(para_t, 3);

                    pcl::PointXYZI pointSel;
                    std::vector<int> pointSearchInd;
                    std::vector<float> pointSearchSqDis;

                    TicToc t_data;
                    // find correspondence for corner features
                    for (int i = 0; i < cornerPointsSharpNum; ++i)
                    {
                        TransformToStart(&(cornerPointsSharp->points[i]), &pointSel);
                        kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                        int closestPointInd = -1, minPointInd2 = -1;
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                        {
                            closestPointInd = pointSearchInd[0];
                            int closestPointScanID = int(laserCloudCornerLast->points[closestPointInd].intensity);

                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;
                            // search in the direction of increasing scan line
                            for (int j = closestPointInd + 1; j < (int)laserCloudCornerLast->points.size(); ++j)
                            {
                                // if in the same scan line, continue
                                if (int(laserCloudCornerLast->points[j].intensity) <= closestPointScanID)
                                    continue;

                                // if not in nearby scans, end the loop
                                if (int(laserCloudCornerLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                    (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                        (laserCloudCornerLast->points[j].z - pointSel.z);

                                if (pointSqDis < minPointSqDis2)
                                {
                                    // find nearer point
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }

                            // search in the direction of decreasing scan line
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if in the same scan line, continue
                                if (int(laserCloudCornerLast->points[j].intensity) >= closestPointScanID)
                                    continue;

                                // if not in nearby scans, end the loop
                                if (int(laserCloudCornerLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                    (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                        (laserCloudCornerLast->points[j].z - pointSel.z);

                                if (pointSqDis < minPointSqDis2)
                                {
                                    // find nearer point
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }
                        }
                        if (minPointInd2 >= 0) // both closestPointInd and minPointInd2 is valid
                        {
                            Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x,
                                                       cornerPointsSharp->points[i].y,
                                                       cornerPointsSharp->points[i].z);
                            Eigen::Vector3d last_point_a(laserCloudCornerLast->points[closestPointInd].x,
                                                         laserCloudCornerLast->points[closestPointInd].y,
                                                         laserCloudCornerLast->points[closestPointInd].z);
                            Eigen::Vector3d last_point_b(laserCloudCornerLast->points[minPointInd2].x,
                                                         laserCloudCornerLast->points[minPointInd2].y,
                                                         laserCloudCornerLast->points[minPointInd2].z);

                            double s;
                            if (DISTORTION)
                                s = (cornerPointsSharp->points[i].intensity - int(cornerPointsSharp->points[i].intensity)) / SCAN_PERIOD;
                            else
                                s = 1.0;
                            ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                            auto ID = problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                            psrIDs.push_back(ID);
                            corner_correspondence++;
                        }
                    }

                    // find correspondence for plane features
                    for (int i = 0; i < surfPointsFlatNum; ++i)
                    {
                        TransformToStart(&(surfPointsFlat->points[i]), &pointSel);
                        kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                        int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                        {
                            closestPointInd = pointSearchInd[0];

                            // get closest point's scan ID
                            int closestPointScanID = int(laserCloudSurfLast->points[closestPointInd].intensity);
                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;

                            // search in the direction of increasing scan line
                            for (int j = closestPointInd + 1; j < (int)laserCloudSurfLast->points.size(); ++j)
                            {
                                // if not in nearby scans, end the loop
                                if (int(laserCloudSurfLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                        (laserCloudSurfLast->points[j].z - pointSel.z);

                                // if in the same or lower scan line
                                if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                // if in the higher scan line
                                else if (int(laserCloudSurfLast->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }

                            // search in the direction of decreasing scan line
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if not in nearby scans, end the loop
                                if (int(laserCloudSurfLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                        (laserCloudSurfLast->points[j].z - pointSel.z);

                                // if in the same or higher scan line
                                if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                else if (int(laserCloudSurfLast->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    // find nearer point
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }

                            if (minPointInd2 >= 0 && minPointInd3 >= 0)
                            {

                                Eigen::Vector3d curr_point(surfPointsFlat->points[i].x,
                                                            surfPointsFlat->points[i].y,
                                                            surfPointsFlat->points[i].z);
                                Eigen::Vector3d last_point_a(laserCloudSurfLast->points[closestPointInd].x,
                                                                laserCloudSurfLast->points[closestPointInd].y,
                                                                laserCloudSurfLast->points[closestPointInd].z);
                                Eigen::Vector3d last_point_b(laserCloudSurfLast->points[minPointInd2].x,
                                                                laserCloudSurfLast->points[minPointInd2].y,
                                                                laserCloudSurfLast->points[minPointInd2].z);
                                Eigen::Vector3d last_point_c(laserCloudSurfLast->points[minPointInd3].x,
                                                                laserCloudSurfLast->points[minPointInd3].y,
                                                                laserCloudSurfLast->points[minPointInd3].z);

                                double s;
                                if (DISTORTION)
                                    s = (surfPointsFlat->points[i].intensity - int(surfPointsFlat->points[i].intensity)) / SCAN_PERIOD;
                                else
                                    s = 1.0;
                                ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                                auto ID = problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                psrIDs.push_back(ID);
                                plane_correspondence++;
                            }
                        }
                    }

                    //printf("coner_correspondance %d, plane_correspondence %d \n", corner_correspondence, plane_correspondence);
                    printf("data association time %f ms \n", t_data.toc());

                    if ((corner_correspondence + plane_correspondence) < 10)
                    {
                        printf("less correspondence! *************************************************\n");
                    }

                    TicToc t_solver;
                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::DENSE_QR;
                    options.max_num_iterations = 4;
                    options.minimizer_progress_to_stdout = false;
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
                    printf("solver time %f ms \n", t_solver.toc());

                    if(opti_counter == 1){
                    //calculate the residuals

                    /*ceres::Problem::EvaluateOptions EvalOpts;
                    EvalOpts.apply_loss_function = false;
                    //EvalOpts.residual_blocks = psrIDs;
                    std::vector<double> Residuals;
                    problem.Evaluate(EvalOpts, nullptr, &Residuals, nullptr, nullptr);
                    double num_res = problem.NumResiduals();
                    double squared_error = std::inner_product( Residuals.begin(), Residuals.end(), Residuals.begin(), 0);
                    //the root mean squared error
                    //double vairance_factor = std::sqrt(squared_error / (num_res - 6.0));
                    double vairance_factor = squared_error / (num_res - 6.0);

                    //int n_res = Residuals.size();
                    //std::cout<<"The number of residual error is:"<<num_res<<std::endl;
                    std::cout<<"The vairance factor is:"<<vairance_factor<<std::endl;
                    //std::cout<<"The number of residual error is:"<<n_res<<std::endl;*/

                 // get the estimation covariance matrix to model the uncertainty
                    ceres::Covariance::Options cov_options;
                    ceres::Covariance covariance(cov_options);

                    vector<pair<const double*, const double*> > covariance_blocks;
                    covariance_blocks.push_back(make_pair(para_q, para_q));
                    covariance_blocks.push_back(make_pair(para_t, para_t));
                    covariance_blocks.push_back(make_pair(para_t, para_q));
                    covariance_blocks.push_back(make_pair(para_q, para_t));

                    CHECK(covariance.Compute(covariance_blocks, &problem));

                    double covariance_qq[4 * 4];
                    double covariance_tt[3 * 3];
                    double covariance_tq[3 * 4];
                    double covariance_qt[4 * 3];

                    covariance.GetCovarianceBlock(para_q, para_q, covariance_qq);
                    covariance.GetCovarianceBlock(para_t, para_t, covariance_tt);
                    covariance.GetCovarianceBlock(para_t, para_q, covariance_tq);
                    covariance.GetCovarianceBlock(para_q, para_t, covariance_qt);



                    if(covariance.GetCovarianceBlock(para_q, para_q, covariance_qq) && covariance.GetCovarianceBlock(para_t, para_t, covariance_tt)&& covariance.GetCovarianceBlock(para_t, para_q, covariance_tq)&& covariance.GetCovarianceBlock(para_q, para_t, covariance_qt))
                    {

                        Eigen::Map<Eigen::Matrix4d> rotation_cov(covariance_qq);
                        Eigen::Map<Eigen::Matrix3d> translation_cov(covariance_tt);
                        Eigen::Map<Eigen::Matrix<double, 3, 4>> tq_cov(covariance_tq);
                        Eigen::Map<Eigen::Matrix<double, 4, 3>> qt_cov(covariance_qt);

                        Eigen::Matrix<double, 7, 7> quat3D_cov;
                        quat3D_cov.topLeftCorner(3, 3) =  translation_cov;
                        quat3D_cov.bottomRightCorner(4, 4) =  rotation_cov;
                        quat3D_cov.block <3, 4>(0, 3) =  tq_cov;
                        quat3D_cov.block <4, 3>(3, 0) =  qt_cov;

                        string filename = "/home/songming/lodometry_cov.txt";
                        ofstream f11;
                        f11.open(filename.c_str(),ios::app);
                        f11 << fixed;
                        f11 << setprecision(9) <<
                        quat3D_cov(0,0) << " " << quat3D_cov(0,1)  << " " << quat3D_cov(0,2) << " "  << quat3D_cov(0,3) << " " << quat3D_cov(0,4)  << " " << quat3D_cov(0,5) << " "  << quat3D_cov(0,6) << " "<<
                        quat3D_cov(1,0) << " " << quat3D_cov(1,1)  << " " << quat3D_cov(1,2) << " "  << quat3D_cov(1,3) << " " << quat3D_cov(1,4)  << " " << quat3D_cov(1,5) << " "  << quat3D_cov(1,6) << " "<<
                        quat3D_cov(2,0) << " " << quat3D_cov(2,1)  << " " << quat3D_cov(2,2) << " "  << quat3D_cov(2,3) << " " << quat3D_cov(2,4)  << " " << quat3D_cov(2,5) << " "  << quat3D_cov(2,6) << " "<<
                        quat3D_cov(3,0) << " " << quat3D_cov(3,1)  << " " << quat3D_cov(3,2) << " "  << quat3D_cov(3,3) << " " << quat3D_cov(3,4)  << " " << quat3D_cov(3,5) << " "  << quat3D_cov(3,6) << " "<<
                        quat3D_cov(4,0) << " " << quat3D_cov(4,1)  << " " << quat3D_cov(4,2) << " "  << quat3D_cov(4,3) << " " << quat3D_cov(4,4)  << " " << quat3D_cov(4,5) << " "  << quat3D_cov(4,6) << " "<<
                        quat3D_cov(5,0) << " " << quat3D_cov(5,1)  << " " << quat3D_cov(5,2) << " "  << quat3D_cov(5,3) << " " << quat3D_cov(5,4)  << " " << quat3D_cov(5,5) << " "  << quat3D_cov(5,6) << " "<<
                        quat3D_cov(6,0) << " " << quat3D_cov(6,1)  << " " << quat3D_cov(6,2) << " "  << quat3D_cov(6,3) << " " << quat3D_cov(6,4)  << " " << quat3D_cov(6,5) << " "  << quat3D_cov(6,6) << std::endl;
                        f11.close();
                        //std::cout<<"The covariance is estimated in frame: "<< framenum<<std::endl;
                        //std::cout<<"The quaternion rotation covariance is estimated as: "<<std::endl<< rotation_cov<<std::endl;
                        //std::cout<<"The translation covariance is estimated as: "<<std::endl<< translation_cov<<std::endl;
                        //std::cout<<"---------------------------------"<<std::endl;

                    }
                    else
                    {
                        string filename = "/home/songming/lodometry_cov.txt";
                        ofstream f22;
                        f22.open(filename.c_str(),ios::app);
                        f22 << fixed;
                        f22 << setprecision(9) <<
                        1e9 << " " << 1e9 << " " << 1e9 << " "  << 1e9 << " " << 1e9 << " " << 1e9 << " "  << 1e9 << " "<<
                        1e9 << " " << 1e9  << " " << 1e9 << " "  << 1e9 << " " << 1e9  << " " << 1e9 << " "<< 1e9 << " "<<
                        1e9 << " " << 1e9 << " " << 1e9 << " "  << 1e9 << " " << 1e9  << " " << 1e9 << " " << 1e9 << " "<<
                        1e9 << " " << 1e9  << " " << 1e9 << " "  << 1e9 << " " << 1e9 << " " << 1e9 << " " << 1e9 << " "<<
                        1e9 << " " << 1e9  << " " << 1e9 << " "  << 1e9 << " " << 1e9  << " " << 1e9 << " " << 1e9 << " "<<
                        1e9 << " " << 1e9  << " " <<1e9 << " "  << 1e9 << " " << 1e9  << " " << 1e9 << " " << 1e9 << " "<<
                        1e9 << " " << 1e9 << " " << 1e9 << " "  << 1e9 << " " << 1e9  << " " << 1e9 << " " << 1e9 << std::endl;
                        f22.close();
                    }

                }
                }
                printf("optimization twice time %f \n", t_opt.toc());

                t_w_curr = t_w_curr + q_w_curr * t_last_curr;
                q_w_curr = q_w_curr * q_last_curr;
                Eigen::Matrix3d R = q_w_curr.normalized().toRotationMatrix();
                //to do, transform to left_cam frame
               // Eigen::Matrix3d R_rel = q_last_curr.normalized().toRotationMatrix();
                //std::cout << "The odometry rotation is: "<< R << std::endl;

                //save the odometry pose in KITTI dataset format


                string filename111 = "/home/songming/lidarrelodometry.txt";
                ofstream f44;
                f44.open(filename111.c_str(),ios::app);
                f44 << fixed;
                f44 << setprecision(9) << t_last_curr[0] << " " << t_last_curr[1]  << " " << t_last_curr[2] << " " <<
                                        q_last_curr.w()<< " " << q_last_curr.x()  << " " << q_last_curr.y() << " "  << q_last_curr.z() << std::endl;
                f44.close();


                string filename = "/home/songming/lidarabsodometry.txt";
                ofstream f;
                f.open(filename.c_str(),ios::app);
                f << fixed;
                f << setprecision(9) << R(0,0) << " " << R(0,1)  << " " << R(0,2) << " "  << t_w_curr(0) << " " <<
                                        R(1,0) << " " << R(1,1)  << " " << R(1,2) << " "  << t_w_curr(1) << " " <<
                                        R(2,0) << " " << R(2,1)  << " " << R(2,2) << " "  << t_w_curr(2) << std::endl;
                f.close();

            }

            TicToc t_pub;

            // publish odometry
            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = "/camera_init";
            laserOdometry.child_frame_id = "/laser_odom";
            laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
            laserOdometry.pose.pose.orientation.x = q_w_curr.x();
            laserOdometry.pose.pose.orientation.y = q_w_curr.y();
            laserOdometry.pose.pose.orientation.z = q_w_curr.z();
            laserOdometry.pose.pose.orientation.w = q_w_curr.w();
            laserOdometry.pose.pose.position.x = t_w_curr.x();
            laserOdometry.pose.pose.position.y = t_w_curr.y();
            laserOdometry.pose.pose.position.z = t_w_curr.z();
            pubLaserOdometry.publish(laserOdometry);

            geometry_msgs::PoseStamped laserPose;
            laserPose.header = laserOdometry.header;
            laserPose.pose = laserOdometry.pose.pose;
            laserPath.header.stamp = laserOdometry.header.stamp;
            laserPath.poses.push_back(laserPose);
            laserPath.header.frame_id = "/camera_init";
            pubLaserPath.publish(laserPath);

            // transform corner features and plane features to the scan end point
            if (0)
            {
                int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
                for (int i = 0; i < cornerPointsLessSharpNum; i++)
                {
                    TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
                }

                int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
                for (int i = 0; i < surfPointsLessFlatNum; i++)
                {
                    TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
                }

                int laserCloudFullResNum = laserCloudFullRes->points.size();
                for (int i = 0; i < laserCloudFullResNum; i++)
                {
                    TransformToEnd(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
                }
            }

            pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
            cornerPointsLessSharp = laserCloudCornerLast;
            laserCloudCornerLast = laserCloudTemp;

            laserCloudTemp = surfPointsLessFlat;
            surfPointsLessFlat = laserCloudSurfLast;
            laserCloudSurfLast = laserCloudTemp;

            laserCloudCornerLastNum = laserCloudCornerLast->points.size();
            laserCloudSurfLastNum = laserCloudSurfLast->points.size();

            // std::cout << "the size of corner last is " << laserCloudCornerLastNum << ", and the size of surf last is " << laserCloudSurfLastNum << '\n';

            kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
            kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

            if (frameCount % skipFrameNum == 0)
            {
                frameCount = 0;

                sensor_msgs::PointCloud2 laserCloudCornerLast2;
                pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
                laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudCornerLast2.header.frame_id = "/camera";
                pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

                sensor_msgs::PointCloud2 laserCloudSurfLast2;
                pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
                laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudSurfLast2.header.frame_id = "/camera";
                pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

                sensor_msgs::PointCloud2 laserCloudFullRes3;
                pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
                laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudFullRes3.header.frame_id = "/camera";
                pubLaserCloudFullRes.publish(laserCloudFullRes3);
            }
            printf("publication time %f ms \n", t_pub.toc());
            printf("whole laserOdometry time %f ms \n \n", t_whole.toc());
            if(t_whole.toc() > 100)
                ROS_WARN("odometry process over 100ms");

            frameCount++;
            framenum++;
        }
        rate.sleep();
    }
    return 0;
}
