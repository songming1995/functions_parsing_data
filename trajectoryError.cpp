#include <iostream>
#include <fstream>
#include <unistd.h>
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>
#include <iomanip>



using namespace Sophus;
using namespace std;

string groundtruth_file = "/home/songming/ORB_SLAM2/GT/03.txt";
string estimated_file =  "/home/songming/ORB_SLAM2/SLAM MODE/FrameTrajectory_KITTI_Format03.txt";


// "/home/songming/ORB_SLAM2/Localization MODE/FrameTrajectory_KITTI_Format03.txt"
//"/home/songming/ORB_SLAM2/SLAM MODE/FrameTrajectory_KITTI_Format03.txt";
//"/home/songming/ORB_SLAM2/GT/03.txt"
typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;

void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti);

TrajectoryType ReadTrajectory(const string &path);

int main(int argc, char **argv)
{
  TrajectoryType groundtruth = ReadTrajectory(groundtruth_file);
  TrajectoryType estimated = ReadTrajectory(estimated_file);
  assert(!groundtruth.empty() && !estimated.empty());
  assert(groundtruth.size() == estimated.size());


  string filename = "/home/songming/ORB_SLAM2/error.txt";

  cout << endl << "Saving Lie algebra error to " << filename << " ..." << endl;
  ofstream f;
  f.open(filename.c_str(),ios::app);
  f << fixed;

  // compute rmse
  double rmse = 0;
  for (size_t i = 0; i < estimated.size(); i++)
  {
    Sophus::SE3d p1 = estimated[i], p2 = groundtruth[i];
    double error = (p2.inverse() * p1).log().norm();
    rmse += error * error;

    auto Lie_error= (p2.inverse() * p1).log();//translation + rotation in sophus
    f << setprecision(9) << Lie_error(3)  << " " << Lie_error(4)  << " "
      << Lie_error(5)  << " " << Lie_error(0)  << " " << Lie_error(1)  << " " << Lie_error(2)  << " " << endl;
  }

  f.close();
  rmse = rmse / double(estimated.size());
  rmse = sqrt(rmse);
  cout << "RMSE = " << rmse << endl;

  DrawTrajectory(groundtruth, estimated);
  return 0;
}

TrajectoryType ReadTrajectory(const string &path) {
  ifstream fin(path);
  TrajectoryType trajectory;
  if (!fin) {
    cerr << "trajectory " << path << " not found." << endl;
    return trajectory;
  }

  while (!fin.eof()) {
    double r00, r01, r02, t0, r10, r11, r12, t1,r20, r21, r22, t2;
    fin >> r00 >> r01 >> r02 >> t0 >> r10 >> r11 >> r12 >> t1 >> r20>>r21>>r22>>t2;
    Matrix3d R ;
    R << r00, r01, r02,
         r10, r11, r12,
         r20, r21, r22;
    Eigen::Quaterniond q(R);
    Sophus::SE3d p1(Eigen::Quaterniond(q), Eigen::Vector3d(t0, t1, t2));
    trajectory.push_back(p1);
  }
  return trajectory;
}

void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti) {
  // create pangolin window and plot the trajectory
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
      pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
  );

  pangolin::View &d_cam = pangolin::CreateDisplay()
      .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
      .SetHandler(new pangolin::Handler3D(s_cam));


  while (pangolin::ShouldQuit() == false) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_cam.Activate(s_cam);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    glLineWidth(2);
    for (size_t i = 0; i < gt.size() - 1; i++) {
      glColor3f(0.0f, 0.0f, 1.0f);  // blue for ground truth
      glBegin(GL_LINES);
      auto p1 = gt[i], p2 = gt[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }

    for (size_t i = 0; i < esti.size() - 1; i++) {
      glColor3f(1.0f, 0.0f, 0.0f);  // red for estimated
      glBegin(GL_LINES);
      auto p1 = esti[i], p2 = esti[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }
    pangolin::FinishFrame();
    usleep(5000);   // sleep 5 ms
  }

}
