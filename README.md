# pypc2map

 Pypc2map is a package that use PointCloud2 to create a global map while scanning the area.
 Package created to be use in replacement of lio_sam mapping global_map.

# Dependencies 

**Pypc2map** catkin dependencies while created the package

     sensor_msg      #Dependencie for subscribe the PointCloud2 
     nav_msgs        #Dependencie for publishing the OccupancyGrid
     rospy

# Requirements
 
 To be able to use the package it will be also needed to install Numpy and Scipy   #if not already  installed on the system .

**Individual installation** 

     pip install numpy
     pip install scipy

**Requirement txt installation**

     pip install -r requirements.txt

# Lauching Pypc2map

     roslaunch pypc2map pypc2map.launch
    




