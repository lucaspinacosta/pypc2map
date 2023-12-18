#!/usr/bin/env python3

"""
The starting point for this script were some functions used to implement 
the progressive morphological filter algorithm (Zhang et al., 2003) to 
generate a raster surface which can be used to classify the ground returns.
This has been updated with a loosely inspired version of the simple 
morphological filter (SMRF) algorithm (Pingel et al., 2013). Nevertheless,
unlike these algorithms that tend to be used to split ground from non-ground 
points within a given pointcloud, it identifies relevant obstacles within the
non-ground points, ultimately generating an occupancy grid (binary image) with
obstacle-free and obstacle cells.

Zhang, K., Chen, S., Whitman, D., Shyu, M., Yan, J., & Zhang, C. (2003). 
A progressive morphological filter for removing nonground measurements 
from airborne LIDAR data. IEEE Transactions on Geoscience and Remote Sensing, 
41(4), 872-882. 

Pingel, T. J., Clarke, K. C., & McBride, W. A. (2013). An improved simple 
morphological filter for the terrain classification of airborne LIDAR data. 
ISPRS Journal of Photogrammetry and Remote Sensing, 77, 21-30.
"""

import rospy
import open3d
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from tf import TransformListener, LookupException, ExtrapolationException, ConnectivityException
from tf.transformations import euler_from_quaternion, quaternion_from_euler, translation_matrix, quaternion_matrix, translation_from_matrix, quaternion_from_matrix
from scipy import ndimage
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
# from std_msgs.msg import Header


class PyPc2Map():
    _data = None
    _frame = ""
    _new_data = False
    
    def __init__(self):
        # relevant inputs, which should be defined as arguments of the launch file
        self.draw_unknown = rospy.get_param('~draw_unknown', True)  # if enabled draws unknown regions, otherwise considers them as known areas
        self.global_frame = rospy.get_param('~global_frame', "odom")
        self.size_offset_x = rospy.get_param('~size_offset_x', 0.0)
        self.size_offset_y = rospy.get_param('~size_offset_y', 0.0)
        self.rolling_window = rospy.get_param('~rolling_window', False)
        self.max_cloud_range = rospy.get_param('~max_cloud_range', 10.0)
        self.resolution = rospy.get_param('~map_resolution', 0.5)   # resolution of the map in meters/pixel
        self.initWinSize = rospy.get_param('~initial_winSize', 1)   # initial window size for morphological opening
        self.maxWinSize = rospy.get_param('~max_winSize', 16)       # maximum window size for morphological opening
        self.winSizeInc = rospy.get_param('~inc_winSize', 1)        # incremental window size for morphological opening
        self.slope = rospy.get_param('~slope', 0.3)                 # slope within the scene
        self.dhmin = rospy.get_param('~min_heigth_dif', 0.6)        # minimum height difference threshold for differentiating ground
        self.dhmax = rospy.get_param('~max_heigth_dif', 1.5)        # maximum height difference threshold for differentiating ground
        self.filters = rospy.get_param('~filters', True)            # point cloud outlier removal filters
        self.filter_rro_nb_points = rospy.get_param('~filter_rro_nb_points', 16) # minimum amount of points that the sphere should contain
        self.filter_rro_radius = rospy.get_param('~filter_rro_radius', 1.0)             # radius of the sphere that will be used for counting the neighbors
        self.filter_rso_nb_neighbors = rospy.get_param('~filter_rso_nb_neighbors', 20)  # how many neighbors are taken into account in order to calculate the average distance for a given point
        self.filter_rso_std_ratio = rospy.get_param('~filter_rso_std_ratio', 5.5)       # setting the threshold level based on the standard deviation of the average distances across the point cloud
        self.frequency = rospy.get_param('~max_operating_frequency', 5) # max node operating frequency. Actual frequency depends on size and frequency of the input cloud.
        self.win_size_mode = rospy.get_param('~win_size_mode', "linear") # max node operating frequency. Actual frequency depends on size and frequency of the input cloud.
        self.draw_mask = rospy.get_param('~draw_mask', True)
        self.size_around_m = rospy.get_param('~size_around_m', 0.8) 
        self.pos =  np.zeros(2, dtype=int)

        rospy.Subscriber('points',PointCloud2,self.pointcloud_callback,queue_size=1)
        self.map_pub = rospy.Publisher('pypc2map',OccupancyGrid,queue_size=1)
        # self.pc_pub = rospy.Publisher('debug_cloud', PointCloud2, queue_size=10)


    def pc2map(self, res, initWinSize, maxWinSize, winSizeInc, slope, dhmin, dhmax):

        # return None if there is not any new data
        if not self._new_data:
            return None

        # reset new data flag !
        self._new_data = False

        t0 = rospy.Time.now().to_sec()

        # load a pointcloud - replace this by subscribing to a pointcloud2 message
        pcd = self._data
        dataArr = np.asarray(pcd)
        if self.filters == True:
            pcd_rad, ind_rad = pcd.remove_radius_outlier(self.filter_rro_nb_points, self.filter_rro_radius)
            pcd_stat, ind_stat = pcd_rad.remove_statistical_outlier(self.filter_rso_nb_neighbors, self.filter_rso_std_ratio)
            dataArr = np.asarray(pcd_stat.points)
        
        # value = 1000.0
        # np.append(dataArr, [value, -value, 0.0])
        # np.append(dataArr, [value, value, 0.0])
        # np.append(dataArr, [-value, value, 0.0])
        # np.append(dataArr, [-value, -value, 0.0])

        # # filter points that violate the defined thresholds
        # tmp = []
        # for p in dataArr:
        #     dist = p[0]**2 + p[1]**2 + p[2]**2
        #     if self.min_cloud_range**2 < dist < self.max_cloud_range**2:
        #         tmp.append(p)

        # dataArr = np.asarray(tmp)
        
        # # DEBUG ONLY
        # header = Header(frame_id="velodyne")
        # msg = pc2.create_cloud_xyz32(header, dataArr)
        # self.pc_pub.publish(msg)
        
        # this is the origin of the map, which should be used to generate the occupancy_map message
        self.originMap = np.array((-self.max_cloud_range, -self.max_cloud_range))
        if not self.rolling_window:
            self.originMap = np.min(dataArr, axis = 0)[0:2]
            self.originMap[0] -= self.size_offset_x
            self.originMap[1] -= self.size_offset_y

        # this is the size of the map, which should be used to generate the occupancy_map message
        sizeMap = np.array(((np.max(dataArr[:,1]) + self.size_offset_y - self.originMap[1])/res + 1,
                            (np.max(dataArr[:,0]) + self.size_offset_x - self.originMap[0])/res + 1)).astype(int)

        # call function to generate the minimum elevation map
        img = self.minElevMap(dataArr, sizeMap, res)

        # segment surface map by applying a morphological opening operation on the minimum surface map
        imgOpen = self.doOpening(img, maxWinSize ,winSizeInc, res, slope, dhmin, dhmax)

        # segment original point cloud and generate related occupancy map, with 0 representing free areas, and 100 representing occupied areas
        imgNew = self.segSurfaceMap(imgOpen, dataArr, sizeMap, res, dhmin, dhmax)

        # find unknown regions and replace related pixels by -1
        if self.draw_unknown:
            [idx, idy] = np.where(np.isinf(imgOpen))
            imgNew[idx, idy] = -1

        rospy.logdebug("Time elapsed: {}".format(rospy.Time.now().to_sec() - t0))

        # find the last positions and replace related pixels by 1 around them
        if self.draw_mask:
            listener = TransformListener()
            try:
                listener.waitForTransform("odom", "bobcat_base", rospy.Time(0), rospy.Duration(3.0))
                (trans, rot) = listener.lookupTransform("odom", "bobcat_base", rospy.Time(0))
                self.pos = np.vstack((self.pos, np.array([trans[0],trans[1]])))
                for _pos in self.pos:
                    idx = np.array((_pos[1]-self.originMap[1])/res).astype(int),np.array((_pos[0]-self.originMap[0])/res).astype(int)
                    size_around = np.array((self.size_around_m)/res).astype(int)
                    x = np.arange(idx[0] - size_around, idx[0] + size_around+1, 1, dtype=int)
                    y = np.arange(idx[1] - size_around, idx[1] + size_around+1, 1, dtype=int)
                    for i in x:
                        for j in y:
                            imgNew[i, j] = 1
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                    rospy.logwarn_throttle(1.0, "{}".format(e))  

        return [self.originMap,imgNew,sizeMap]


    def elevationDiffTreshold(self, c, wk, wk1, s, dh0, dhmax):
        """
        Function to determine the elevation difference threshold based on window size (wk)
        c is the bin size is metres. Default values for site slope (s), initial elevation 
        differents (dh0), and maximum elevation difference (dhmax). These will change 
        based on environment.
        """
    
        if wk <= 3:
            dht = dh0
        elif wk > 3:
            dht = s * (wk-wk1) * c + dh0
        
        #However, if the difference threshold is greater than the specified max threshold, 
        #set the difference threshold equal to the max threshold    
        if dht > dhmax:
            dht == dhmax
            
        return dht                                

    def disk(self,radius, dtype=np.uint8):
        """
        Generates a flat, disk-shaped structuring element.
        A pixel is within the neighborhood if the euclidean distance between
        it and the origin is no greater than radius.
        Parameters:

        * radius : int The radius of the disk-shaped structuring element.

        Other Parameters:

        * dtype : data-type The data type of the structuring element.

        Returns:

        * selem : ndarray The structuring element where elements of the neighborhood
        are 1 and 0 otherwise.

        """
        L = np.arange(-radius, radius + 1)
        X, Y = np.meshgrid(L, L)
        return np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)

    def doOpening(self,iarray, maxWindowSize, winSizeInc, c, s, dh0, dhmax):
        """
        A function to perform a series of iterative opening operations
        on the data array with increasing window sizes. 
        """
        
        # calculate parameter k (array of values to increment the window size) k = 1,2,..., M
        k = np.arange(0, maxWindowSize, winSizeInc)

        # calculate the (real) maximum windows size
        if self.win_size_mode == "linear":
            #print("Maximum windows size: linear")
            winSize = (2*k*1) + 1
        elif self.win_size_mode == "exp":
            winSize = (2**k) + 1
        else:
            winSize = (2*k*1) + 1   # other possibilies

        # create array of each window size's previous window size (i.e. window size at t-1)
        winSize1 = np.zeros([winSize.shape[0]])
        winSize1[1:] = winSize[:-1]
        
        wkIdx = 0
        for wk in winSize1:
            #print(wk)
            if wk <= maxWindowSize:
                if wkIdx > 0:
                    wk1 = winSize1[wkIdx-1]
                    s = dhmax/((wk - wk1)/2)
                else:
                    wk1 = 0    
                dht = self.elevationDiffTreshold(c, wk, wk1, s, dh0, dhmax)       
                Z = iarray
                
                structureElement = self.disk(wk)
                Zf = ndimage.grey_opening(Z, structure=structureElement, size=structureElement.shape)
                
                #Trying new method - only replace the value if it's less than the specified height
                #threshold or the zalue is less than the input
                zDiff = np.absolute(Z - Zf)
                iarray = np.where(np.logical_or(zDiff<=dht,Zf<Z), Zf, Z) # np.where(np.logical_or(zDiff<=dht,Zf<Z), Zf, Z)                     
                wkIdx += 1
        return iarray 

    def minElevMap(self,dataArr, sizeMap, res):
        """
        A function to generate a minimum elevation map. 
        """
        # divide the point cloud data into grids along the xy- plane (bird's-eye view)
        img = np.full(sizeMap, np.inf)
        
        for dataArri in dataArr:
            # find the lowest elevation value for each grid element (pixel)
            idx = np.array((dataArri[1]-self.originMap[1])/res).astype(int),np.array((dataArri[0]-self.originMap[0])/res).astype(int)
            if img[idx] > dataArri[2]:
                # combine all the Zmin values into a 2-D matrix (raster image) to create a minimum elevation surface map
                img[idx] = dataArri[2]
        return img

    def segSurfaceMap(self,imgOpen, dataArr, sizeMap, res, dhmin, dhmax):
        """
        A function to segment a point cloud and generate related occupancy map, 
        with 0 representing free areas, and 100 representing occupied areas
        """
        imgNew = np.zeros(sizeMap)
        for dataArri in dataArr:
            idx = np.array((dataArri[1]-self.originMap[1])/res).astype(int), np.array((dataArri[0]-self.originMap[0])/res).astype(int)
            if imgOpen[idx] + dhmin < dataArri[2] <= imgOpen[idx] + dhmax:
                imgNew[idx] = 100
        return imgNew

    def pointcloud_callback(self, msg):
        data = np.array(list(pc2.read_points(
            msg, skip_nans=True, field_names=['x', 'y', 'z'])))

        if self.filters:
            tmp = open3d.geometry.PointCloud()
            tmp.points = open3d.utility.Vector3dVector(data)
            data = tmp

        self._data = data
        self._frame = msg.header.frame_id
        self._new_data = True

    def main(self):
        listener = TransformListener()
        rate = rospy.Rate(self.frequency)
        while not rospy.is_shutdown():
            rate.sleep()

            try:
                mapPc2 = self.pc2map(self.resolution, self.initWinSize, self.maxWinSize,
                                 self.winSizeInc, self.slope, self.dhmin, self.dhmax)
            except:
                rospy.logwarn("Initial cloud does not have enough points.")

            # jump iteration if mapPc2 is invalid
            if not mapPc2:
                continue

            # fetch the transform between the global frame and the cloud frame
            pos = mapPc2[0][0:2]
            rot = np.array((0, 0, 0, 1), dtype=np.float64)
            if self.rolling_window:
                try:
                    listener.waitForTransform(self.global_frame, self._frame, rospy.Time(0), rospy.Duration(3.0))
                    (trans, rot) = listener.lookupTransform(self.global_frame, self._frame, rospy.Time(0))

                    # shift origin to center the map in the base link frame
                    local_center = np.dot(translation_matrix((pos[0], pos[1], .0)), quaternion_matrix((.0, .0, .0, 1.0)))
                    transformation = np.dot(translation_matrix(trans), quaternion_matrix(rot))
                    framed_center = np.dot(transformation, local_center)
                    trans = translation_from_matrix(framed_center)
                    rot = quaternion_from_matrix(framed_center)

                    # set the appropriate centered translation
                    pos[0] = trans[0]
                    pos[1] = trans[1]

                    # negate roll and pitch, only care about the yaw
                    rpy = euler_from_quaternion([rot[0], rot[1], rot[2], rot[3]])
                    rot = quaternion_from_euler(.0, .0, rpy[2])
                except (LookupException, ConnectivityException, ExtrapolationException) as e:
                    rospy.logwarn_throttle(1.0, "{}".format(e))

            grid = OccupancyGrid()
            grid.header.frame_id = self.global_frame
            grid.info.height = mapPc2[2][0]
            grid.info.width = mapPc2[2][1]
            grid.info.resolution = self.resolution
            grid.info.origin.position.x = pos[0]
            grid.info.origin.position.y = pos[1]
            grid.info.origin.position.z = 0.0
            grid.info.origin.orientation.x = rot[0]
            grid.info.origin.orientation.y = rot[1]
            grid.info.origin.orientation.z = rot[2]
            grid.info.origin.orientation.w = rot[3]
            grid.data = np.array(mapPc2[1]).flatten().astype(np.int8).tolist()
            self.map_pub.publish(grid)

        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('pypcMap', anonymous=True)
    pp2m = PyPc2Map()
    pp2m.main()
