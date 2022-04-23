import copy
import numbers
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import open3d as o3d
import random
import json
import scipy.interpolate as si
import matplotlib.colors as colors

try:
    from scipy.spatial import cKDTree as KDTree
except ImportError:
    from scipy.spatial import KDTree

from . import data
from .trirender import UVTrianglesRenderer


def slice_by_plane(mesh, center, n):
    c = np.dot(center, n)
    plane_side = lambda x: np.dot(x, n) >= c
    split = np.asarray([plane_side(v) for v in mesh.vertices])
    slice1_indices = np.argwhere(split == True)
    slice2_indices = np.argwhere(split == False)
    return slice1_indices, slice2_indices


def remove_points(mesh, indices, blackoutTexture=True):
    submesh = data.Mesh()

    roi_vertices = np.ones(len(mesh.vertices), dtype=bool)
    roi_vertices[indices] = False
    submesh.vertices = mesh.vertices[roi_vertices]
    if mesh.vertex_colors is not None:
        submesh.vertex_colors = mesh.vertex_colors[roi_vertices]
    if mesh.vertex_normals is not None:
        submesh.vertex_normals = mesh.vertex_normals[roi_vertices]

    if mesh.faces is not None:
        removed_faces = np.any(np.isin(mesh.faces, indices), axis=1)
        roi_faces = ~removed_faces
        faces_subset = mesh.faces[roi_faces]

        idx_map = -np.ones(len(mesh.vertices), dtype=int)
        idx_map[roi_vertices] = np.arange(sum(roi_vertices))
        faces_subset = idx_map[faces_subset]
        submesh.faces = faces_subset

        if mesh.texture_indices is not None:
            texture_faces_subset = mesh.texture_indices[roi_faces]

            roi_texcoords = np.zeros(len(mesh.texcoords), dtype=bool)
            kept_texcoords_indices = np.unique(texture_faces_subset)
            roi_texcoords[kept_texcoords_indices] = True

            idx_map = -np.ones(len(mesh.texcoords), dtype=int)
            idx_map[roi_texcoords] = np.arange(sum(roi_texcoords))
            texture_faces_subset = idx_map[texture_faces_subset]

            submesh.texture_indices = texture_faces_subset
            submesh.texcoords = mesh.texcoords[roi_texcoords]

            if blackoutTexture:
                tri_indices = submesh.texture_indices
                tex_coords = submesh.texcoords
                img = render_texture(mesh.texture, tex_coords, tri_indices)
                # dilate the result to remove sewing
                kernel = np.ones((3, 3), np.uint8)
                texture_f32 = cv2.dilate(img, kernel, iterations=1)
                submesh.texture = texture_f32.astype(np.float64)

        if mesh.faces_normal_indices is not None and mesh.face_normals is not None:
            normals_faces_subset = mesh.faces_normal_indices[roi_faces]

            roi_normals = np.zeros(len(mesh.face_normals), dtype=bool)
            kept_normals_indices = np.unique(normals_faces_subset)
            roi_normals[kept_normals_indices] = True

            idx_map = -np.ones(len(mesh.face_normals), dtype=int)
            idx_map[roi_normals] = np.arange(sum(roi_normals))
            normals_faces_subset = idx_map[normals_faces_subset]

            submesh.faces_normal_indices = normals_faces_subset
            submesh.face_normals = mesh.face_normals[roi_normals]

    return submesh


def render_texture(texture, tex_coords, tri_indices):
    if len(texture.shape) == 3 and texture.shape[2] == 4:
        texture = texture[:, :, 0:3]
    elif len(texture.shape) == 2:
        texture = np.concatenate([texture, texture, texture], axis=2)

    renderer = UVTrianglesRenderer.with_standalone_ctx(
        (texture.shape[1], texture.shape[0])
    )

    return renderer.render(tex_coords, tri_indices, texture, True)


def estimate_plane(a, b, c):
    """Estimate the parameters of the plane passing by three points.

    Returns:
        center(float): The center point of the three input points.
        normal(float): The normal to the plane.
    """
    center = (a + b + c) / 3
    normal = np.cross(b - a, c - a)
    assert np.isclose(np.dot(b - a, normal), np.dot(c - a, normal))
    return center, normal


def crop_byVisibility(
    inputMesh, nViews=2, spreadViews=4, pLevel=1, mask_faces=None, debug_view=False
):
    def toO3DMesh(m):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(m.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(m.faces)
        mesh.compute_vertex_normals()
        return mesh

    def get_viewPts(m, numViews, debug_view):
        minV = np.min(m.get_min_bound()) * 1.15
        maxV = np.max(m.get_max_bound()) * 0.95
        points = []
        for i in range(numViews):
            y = maxV - (i / float(numViews - 1)) * (maxV - minV)  # y goes from 1 to -1
            radius = (
                math.sqrt(pow(abs(maxV - minV), 2) - pow(y, 2)) * 0.55
            )  # radius at y
            theta = math.pi * (3.0 - math.sqrt(5.0)) * i  # golden angle increment
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            points.append([x, y, z])

        if debug_view:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            m.compute_vertex_normals()
            o3d.visualization.draw_geometries([pcd, m])

        return np.array(points)

    def get_views(viewPoints, debug_view):
        # Generate a set of views to generate depth maps from.
        #:param n_views: number of views per axis
        #:type n_views: int
        #:return: rotation matrices
        #:rtype: [numpy.ndarray]
        Rs = []
        ts = []

        for i in range(viewPoints.shape[0]):
            # https://math.stackexchange.com/questions/1465611/given-a-point-on-a-sphere-how-do-i-find-the-angles-needed-to-point-at-its-ce
            longitude = -math.atan2(viewPoints[i, 0], viewPoints[i, 1])
            latitude = math.atan2(
                viewPoints[i, 2],
                math.sqrt(viewPoints[i, 0] ** 2 + viewPoints[i, 1] ** 2),
            )

            R_x = np.array(
                [
                    [1, 0, 0],
                    [0, math.cos(latitude), -math.sin(latitude)],
                    [0, math.sin(latitude), math.cos(latitude)],
                ]
            )
            R_y = np.array(
                [
                    [math.cos(longitude), 0, math.sin(longitude)],
                    [0, 1, 0],
                    [-math.sin(longitude), 0, math.cos(longitude)],
                ]
            )

            R = R_y.dot(R_x)
            Rs.append(R)
            ts.append(viewPoints[i])

        return viewPoints, Rs, ts

    def bbox_to_Polygon(MaxP, MinP):
        p1 = [MaxP[0], MaxP[1], MinP[2]]
        p2 = [MaxP[0], MinP[1], MinP[2]]
        p3 = [MinP[0], MaxP[1], MinP[2]]
        p4 = [MinP[0], MinP[1], MinP[2]]
        p5 = [MaxP[0], MaxP[1], MaxP[2]]
        p6 = [MinP[0], MaxP[1], MaxP[2]]
        p7 = [MinP[0], MinP[1], MaxP[2]]
        p8 = [MaxP[0], MinP[1], MaxP[2]]

        listPoints = np.asarray([p1, p3, p4, p2, p5, p6, p7, p8]).reshape(8, 3)
        return listPoints

    def mask_faces_from_vertex_indices(faces, indices, mask_faces=None):
        face_indices = np.any(np.isin(faces[:], indices, assume_unique=False), axis=1)
        c_mask_faces = np.ones(len(faces))
        c_mask_faces[face_indices] = 0
        if mask_faces is not None:
            return mask_faces * c_mask_faces
        else:
            return c_mask_faces

    def do_crop(meshIn, mask_faces, nViews, spreadViews, pLevel, debug_view):
        mesh = toO3DMesh(meshIn)
        # Centers the mesh around origin, required to get the view-points
        mesh_center = mesh.get_center()
        mesh.translate(-mesh_center)
        # Build a KD-Tree on Original Mesh
        kdtreeM = KDTree(np.asarray(mesh.vertices), leafsize=3)

        spts, sview_Rs, sview_ts = get_views(
            get_viewPts(mesh, nViews * 4, debug_view), debug_view
        )  # <----- sparse views
        dpts, dview_Rs, dview_ts = get_views(
            get_viewPts(mesh, nViews * 20, debug_view), debug_view
        )  # <----- dense views

        LoLo_Crop_indices = []
        # Take Neighbourhood view points around a
        kdtree = KDTree(dpts, leafsize=3)
        for sv in spts:
            cm = o3d.geometry.TriangleMesh()
            _, indices = kdtree.query(sv, k=spreadViews)
            minbb = mesh.get_min_bound()  # * 1.15
            maxbb = mesh.get_max_bound() * 1.05
            bb_pcd = o3d.geometry.PointCloud()
            bb_pcd.points = o3d.utility.Vector3dVector(bbox_to_Polygon(maxbb, minbb))

            for i in indices:
                i_viewTrans = np.eye(4)
                i_viewTrans[:3, :3] = dview_Rs[i]
                i_viewTrans[:3, 3] = dview_ts[i]
                bb = copy.deepcopy(bb_pcd)
                bb.transform(i_viewTrans)

                mesh_crp = mesh.crop(bb.get_axis_aligned_bounding_box())
                cm += mesh_crp
                cm.remove_duplicated_triangles()
                cm.remove_duplicated_vertices()

            cm.remove_duplicated_triangles()
            cm.remove_duplicated_vertices()

            fA = cm.get_surface_area() / mesh.get_surface_area()

            # Based Area ratio between the Cropped and Original Mesh
            if pLevel == 1:
                if fA > 0.65 and fA < 0.95:
                    cm.paint_uniform_color(
                        [
                            random.uniform(0.0, 0.8),
                            random.uniform(0.5, 1.0),
                            random.uniform(0.0, 0.5),
                        ]
                    )
                    if debug_view == True:
                        o3d.visualization.draw_geometries([cm])
                    print("Cropping Level 1")
                else:
                    print("Cropping Level 1 IGNORED %f" % fA)
                    continue
            elif pLevel == 2:
                if fA > 0.30 and fA < 0.95:
                    cm.paint_uniform_color(
                        [
                            random.uniform(0.0, 0.8),
                            random.uniform(0.5, 1.0),
                            random.uniform(0.0, 0.5),
                        ]
                    )
                    if debug_view == True:
                        o3d.visualization.draw_geometries([cm])
                    print("Cropping Level 2")
                else:
                    print("Cropping Level 2 IGNORED %f" % fA)
                    continue
            elif pLevel == 3:
                if fA > 0.02 and fA < 0.95:
                    cm.paint_uniform_color(
                        [
                            random.uniform(0.0, 0.8),
                            random.uniform(0.5, 1.0),
                            random.uniform(0.0, 0.5),
                        ]
                    )
                    if debug_view == True:
                        o3d.visualization.draw_geometries([cm])
                    print("Cropping Level 3")
                else:
                    print("Cropping Level3 IGNORED %f" % fA)
                    continue
            else:
                continue

                continue

            # NOTE return indices to_crop
            # k=2 is used for safety over degeneracy
            to_crop = []
            for i in range(len(np.asarray(cm.vertices))):
                _, indices = kdtreeM.query(cm.vertices[i], k=2)
                to_crop.append(indices)

            if len(to_crop) > 0:
                to_crop = np.unique(np.concatenate(to_crop))
                LoLo_Crop_indices.append(to_crop)

        return LoLo_Crop_indices

    return do_crop(inputMesh, mask_faces, nViews, spreadViews, pLevel, debug_view)
