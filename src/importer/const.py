"""
 Global constants

 Created by Chen Chen on 07/09/2013
"""
import pyproj

# Display setup
figsize = (16,9)
colors = 'bgrcmyk'
arrow_params = {'length_includes_head':True, 'shape':'full', 'head_starts_at_zero':False, 'zorder':3}

# Beijing pyproj
BJ_utm_projector = pyproj.Proj(proj='utm', zone=50, south=False, ellps='WGS84')
# utm_projector(easting, northing, inverse=True)
# San Francisco
SF_utm_projector = pyproj.Proj(proj='utm', zone=10, south=False, ellps='WGS84')


# All points here are expressed in UTM coordinates (easting, northing)
BJ_SW = (427450, 4404400) # lon: 116.15270638; 
                          # lat:  39.78645910.
BJ_NE = (466500, 4438500) # lon: 116.60699204;
                          # lat:  40.09612282.

# Beijing test regions
""" Ten ( 2km x 2km ) rectangular boxes for testing.
    They overlap by 1km.
"""
BB_SW = [(440000, 4421450), (441000, 4421450), (442000, 4421450), (443000, 4421450),\
         (444000, 4421450), (445000, 4421450), (446000, 4421450), (447000, 4421450),\
         (448000, 4421450), (449000, 4421450)]
BB_NE = [(442000, 4423450), (443000, 4423450), (444000, 4423450), (445000, 4423450),\
         (446000, 4423450), (447000, 4423450), (448000, 4423450), (449000, 4423450),\
         (450000, 4423450), (451000, 4423450)]

TEST_CENTER = [(441144, 4422470),(441160, 4422070),(441166, 4423050),(441688, 4423030),(440511, 4422390),(440535, 4421890),(440761, 4421790),(440505, 4422560),(440235, 4423320),(440239, 4423210),(440239, 4423060),(440438, 4423330),(440485, 4423060),(440498, 4422710),(440515, 4422390),(440538, 4421900),(440096, 4421800),(441678, 4423030),(441795, 4421890),(440844, 4422430),(440927, 4422040),(441180, 4421830),(441362, 4422820),(440109, 4422280),(441652, 4423280),(442698, 4423100),(442765, 4422520),(443629,4423160),(443825,4422280),(443175,4422020),(443954,4421630),(442553,4421670),(443321,4421620),(444106,4423090),(444497,4421580),(444659,4422500),(444937,4421580),(444381,4423140),(444924,4423130)]

CF_ABSOLUTE_TIME_TO_UTC_OFFSET = 28800 # in sec, basically, this is the UTC for 1 Jan 2001 00:00:00 GMT

# Test Region
BJ_TEST_SW = (446000, 4421450) # (116.36792476608906, 39.94144995449493)
BJ_TEST_NE = (451000, 4426450) # (116.426070795357, 39.98680126977899)

# Test San Francisco Region

SF_RANGE_SW = (550000, 4178000)
SF_RANGE_NE = (555000, 4183000)

SF_small_RANGE_SW = (551000, 4180000)
SF_small_RANGE_NE = (553000, 4182000)

# Test location and region radius
R = 500
Region_0_LOC = (447772, 4424300)
Region_1_LOC = (446458, 4422150)
SF_LOC = (551281, 4180430) # San Francisco

# Colors
BLUE = '#6699cc'
GRAY = '#999999'
