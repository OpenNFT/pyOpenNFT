import time

import numpy as np
from scipy.io import savemat

from opennft import prepare_orth_view as pov
from opennft import update_orth_view as uov


def test_orth_vol(data_path, nii_image_1, struct_image, orth_matlab, orth_matlab_struct):

    mat = nii_image_1.affine
    # mat: a 12-parameter affine transform (from sform0)
    #      Note that the mapping is from voxels (where the first
    #      is considered to be at [1,1,1], to millimetres
    mat = mat @ np.hstack((np.eye(4, 3), np.array([-1, -1, -1, 1], ndmin=2).T))

    dim = np. array(nii_image_1.shape)
    vol = np.array(nii_image_1.get_fdata(), order='F')

    str_param1 = pov.prepare_orth_view(mat, dim)

    imgt, imgc, imgs = uov.update_orth_view(str_param1, vol, mat)

    resl_dic = {"imgt_p": imgt, "imgc_p": imgc, "imgs_p": imgs}
    savemat(data_path / "orth_python.mat", resl_dic)

    np.testing.assert_almost_equal(imgt, orth_matlab['imgt'], decimal=0, err_msg="Not equal")
    print("\nimgt delta = %.4f" % ((imgt - orth_matlab['imgt']) ** 2).mean(axis=None))
    np.testing.assert_almost_equal(imgc, orth_matlab['imgc'], decimal=0, err_msg="Not equal")
    print("imgc delta = %.4f" % ((imgc - orth_matlab['imgc']) ** 2).mean(axis=None))
    np.testing.assert_almost_equal(imgs, orth_matlab['imgs'], decimal=0, err_msg="Not equal")
    print("imgs delta = %.4f" % ((imgs - orth_matlab['imgs']) ** 2).mean(axis=None))

    mat = struct_image.affine
    vol = np.array(struct_image.get_fdata(), order='F')

    start = time.time()
    imgt, imgc, imgs = uov.update_orth_view(str_param1, vol, mat)
    print(time.time()-start)

    resl_dic = {"imgt_p_struct": imgt, "imgc_p_struct": imgc, "imgs_p_struct": imgs}
    savemat(data_path / "orth_python_struct.mat", resl_dic)

    np.testing.assert_almost_equal(imgt, orth_matlab_struct['imgt_struct'], decimal=0, err_msg="Not equal")
    print("\nimgt_struct delta = %.4f" % ((imgt - orth_matlab_struct['imgt_struct']) ** 2).mean(axis=None))
    np.testing.assert_almost_equal(imgc, orth_matlab_struct['imgc_struct'], decimal=0, err_msg="Not equal")
    print("imgc_struct delta = %.4f" % ((imgc - orth_matlab_struct['imgc_struct']) ** 2).mean(axis=None))
    np.testing.assert_almost_equal(imgs, orth_matlab_struct['imgs_struct'], decimal=0, err_msg="Not equal")
    print("imgs_struct delta = %.4f" % ((imgs - orth_matlab_struct['imgs_struct']) ** 2).mean(axis=None))

    assert True, "Done"
