#!/usr/bin/env python3

from dis import dis
import shapely.geometry as geometry
import numpy as np
import math
from math import pi
import pandas as pd
import matplotlib.pyplot as plt
from shapely.ops import polygonize, unary_union
from scipy.spatial import Delaunay, ConvexHull
import seaborn as sns
import argparse

def alpha_shape(points, alpha):

    #CREDIT: https://gist.github.com/dwyerk/10561690

    """
    Compute the alpha shape (concave hull) of a set of points.

    @param points: Iterable container of points
    @param alpha: alpha value to influence gooeyness of the border. Smaller
                  numbers don't fall inward as much as larger numbers. Too
                  large, and you lose everything.
    """

    def add_edge(edges, edge_points, coords, i, j):
        """ Add a line between the ith and jth points, if not in list already."""
        if (i, j) in edges or (j, i) in edges:
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])

    # When triangular, no sense in computing the alpha shape
    if len(points) < 4:
        return geometry.MultiPoint(list(points)).convex_hull
    
    coords = points
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # The following is to plot the result of the Delaunay triangulation
    #plt.triplot(coords[:,0], coords[:,1], tri.simplices)
    #plt.plot(coords[:,0], coords[:,1], 'o')
    #plt.show()
    # The following is to plot the result of the convex Hull
    #hull = ConvexHull(points)
    #plt.plot(points[:,0], points[:,1], 'o')
    #for simplex in hull.simplices:
    #    plt.plot(points[simplex,0], points[simplex,1], 'k-')
    #plt.show()
    #plt.savefig("./convex_hull.jpeg", bbox_inches="tight", dpi=300)


    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

        # Semiperimeter of triangle
        s = (a + b + c)/2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        if area <= 0:
            continue
        circum_r = a*b*c/(4.0*area)

        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    polygons = unary_union(triangles)

    return polygons

def plot_data(data, d1, d2, overall_shape, sub_shape):

    color_by = 'dataset'
    df_subset = data.sort_values(by=[color_by], ascending=False)

    sns.scatterplot(
        x=d1, y=d2,
        hue=color_by,
        palette=['grey', 'red'],
        data=df_subset,
        legend=False,
        s=3,
        alpha=1,
        linewidth=0,
        edgecolors=None
    )

    # plot the shapes from all the data
    for num, geom in enumerate(overall_shape.geoms):
        plt.plot(*geom.exterior.xy, color='black')

    # plot the shapes only from our database
    for geom in sub_shape.geoms:
        plt.plot(*geom.exterior.xy, color='blue')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("./concave_hull.jpeg", bbox_inches="tight", dpi=300)

def main(file, d1, d2, subset, a):
    data = pd.read_csv(file)
    data = data[[d1] + [d2] + ['dataset']]
    allpoints = data[[d1, d2]].values
    overall_shape = alpha_shape(allpoints, alpha)
    overall_shape_area = overall_shape.area
    points = data.loc[data["dataset"]== subset][[d1, d2]].values
    sub_shape = alpha_shape(points, alpha)
    sub_shape_area = sub_shape.area

    # Calculate the disparity
    disparity_subset = (sub_shape.area / overall_shape.area)
    # This should be equal to 1
    # disparity_set = (overall_shape.area / overall_shape.area)

    plot_data(data=data, d1=d1, d2=d2, overall_shape=overall_shape, sub_shape=sub_shape)
    
    return sub_shape_area, overall_shape_area, disparity_subset

if __name__ == '__main__':
    fulldescription = """
    Calculate disparity using the alpha shape of 2D data.
    The 2D data can be two descriptors or dimension-reduced data (e.g., UMAP).
    The code takes a csv file as input, and requires the user to specify the two
    column names to use for the alpha shape (e.g., umap1, umap2). Finally, an alpha
    value can be tuned to acquire the desired alpha shape (i.e., the shape which best
    outlines the data). The code will give a png file of the alpha shape for validation.
    Since the calculation of disparity relies on the comparison of a subset to a superset,
    the csv file must label the data according to which dataset it belongs to.
    The name of the subset label must be specified by the user. These labels must
    exist under a column named "dataset". The superset is assumed to be all
    points in the csv file.
    """

    parser = argparse.ArgumentParser(description=fulldescription)
    parser.add_argument('csv_file', type=str,
                        help='CSV file containing the 2D data.')
    parser.add_argument('dim1', type=str,
                        help='The first dimension for the alpha shape calculation.')
    parser.add_argument('dim2', type=str,
                        help='The second dimension for the alpha shape calculation.')
    parser.add_argument('subset', type=str,
                        help='The name of the subset label in the csv file.')
    parser.add_argument('--a', default=1.0, type=float,
                        help='The alpha value to use to compute the alpha shape.')

    args = parser.parse_args()
    csv_file = args.csv_file
    dim1 = args.dim1
    dim2 = args.dim2
    alpha = args.a
    subset = args.subset
    sub_area, overall_area, disparity = main(file=csv_file, d1=dim1, d2=dim2, subset=subset, a=alpha)

    print("Area occupied by subset: {}".format(sub_area))
    print("Area occupied by superset: {}".format(overall_area))
    print("Disparity: {}".format(disparity))