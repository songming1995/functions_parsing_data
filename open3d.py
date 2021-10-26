vis = o3d.visualization.Visualizer()
vis.create_window()

num_iter = 300
for i in range(num_iter):
    print("/home/songming/velodyne_points/data/%010d.bin"%i)
    bin_pcd = np.fromfile("/home/songming/velodyne_points/data/%010d.bin"%i, dtype=np.float32)
    points = bin_pcd.reshape((-1, 4))[:, 0:3]
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    source = o3d_pcd.voxel_down_sample(voxel_size=0.2)
    print(source)
    
    if i == 0:
        vis.add_geometry(source)
        print("initialization finished")
    else:
        vis.clear_geometries()
        vis.add_geometry(source)
        
        print("update finished")
        
    vis.poll_events()
    vis.update_renderer()
    
vis.destroy_window()
