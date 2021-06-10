import numpy as np 
import open3d as o3d

def test():
    loadData_descriptors = np.load('geometric_registration/D3Feat06070021/descriptors/7-scenes-redkitchen/cloud_bin_0.D3Feat.npy')
    print("----type----")
    print(type(loadData_descriptors))
    print("----shape----")
    print(loadData_descriptors.shape)

    loadData_keypoints = np.load('geometric_registration/D3Feat06070021/keypoints/7-scenes-redkitchen/cloud_bin_0.npy')
    print("----type----")
    print(type(loadData_keypoints))
    print("----shape----")
    print(loadData_keypoints.shape)

    loadData_scores = np.load('geometric_registration/D3Feat06070021/scores/7-scenes-redkitchen/cloud_bin_0.npy')
    print("----type----")
    print(type(loadData_scores))
    print("----shape----")
    print(loadData_scores.shape)

def visualize():

    point_file = "7-scenes-redkitchen/cloud_bin_40"
    loadData_scores = np.load('geometric_registration/D3Feat06070021/scores/' + point_file + '.npy')
    print("----shape----")
    print(loadData_scores.shape)
    ind = np.argpartition(-loadData_scores.T, 1000)[0,:1000]
    # ind = np.where(loadData_scores>0.5)[0]
    print(ind.shape)
    print(loadData_scores[ind,0])

    loadData_keypoints = np.load('geometric_registration/D3Feat06070021/keypoints/'+ point_file+'.npy')
    print("----shape----")
    print(loadData_keypoints.shape)

    input_file = "data/3DMatch/fragments/" + point_file + ".ply"
    pcd = o3d.io.read_point_cloud(input_file) # Read the point cloud
    # Convert open3d format to numpy array
    # Here, you have the point cloud in numpy format. 
    loadPointCLoud = np.asarray(pcd.points) 
    print("----shape----")
    print(loadPointCLoud.shape)
    # colors = [[1, 0, 0] for i in range(loadPointCLoud.shape[0])]
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    selected_points = loadData_keypoints[ind,:]
    print(selected_points.shape)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=960) 

    rc = vis.get_render_option()
    rc.point_size = 1

    bbx_points = 0.03*(np.array([0,0,0, 0,1,0, 1,1,0, 1,0,0, 0,0,1, 0,1,1, 1,1,1, 1,0,1]).reshape([8,3]) - np.array([0,0,0.5]))

    for ii in range(selected_points.shape[0]):
        
        bbx_points_ii = bbx_points + selected_points[ii,:]
        lines =  [ [i, i+1]  for i in range(3)]
        lines += [ [i+4, i+5]  for i in range(3)]
        lines += [ [i, i+4]  for i in range(4)]
        lines += [ [3,0], [7,4]]

        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(bbx_points_ii),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)

    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()

def visualize_heatmap():

    point_file = "7-scenes-redkitchen/cloud_bin_0"
    # point_file = "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika/cloud_bin_0"
    loadData_scores = np.load('geometric_registration/D3Feat06070021/scores/' + point_file + '.npy')
    print("----shape----")
    print(loadData_scores.shape)

    loadData_keypoints = np.load('geometric_registration/D3Feat06070021/keypoints/'+ point_file+'.npy')
    print("----shape----")
    print(loadData_keypoints.shape)

    input_file = "data/3DMatch/fragments/" + point_file + ".ply"
    # vis = o3d.visualization.Visualizer()
    # pcd = o3d.io.read_point_cloud(input_file) # Read the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(loadData_keypoints)
    colors = []
    max_score = np.max(loadData_scores)
    min_score = np.min(loadData_scores)
    for score in loadData_scores:
        # (1,1,1) -> (1,0,0)  (1,1,1) -> (0,0,1)
        score = (score-min_score)/(max_score-min_score)
        # score = min(2*score,1)
        if score < 0.5:
            colors.append([1.5*score, 1.5*score,1])
        if score > 0.5:
            colors.append([1,-1.5*score+1.5, -1.5*score+1.5])
    
    print(np.array(colors).shape)    
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=960) 

    rc = vis.get_render_option()
    rc.point_size = 6
    
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

# visualize()
visualize_heatmap()